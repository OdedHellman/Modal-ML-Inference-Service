import time
from dotenv import load_dotenv

load_dotenv()

from io import BytesIO
import os

import modal

IS_COMPILE_MODEL = False
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "fastapi[standard]",
        "python-dotenv",
        "pydantic>=2,<3",
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
        "boto3",
    )
    .env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"}  # Faster downloads
    )
)

flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

flux_image = flux_image.add_local_python_source("schemas", "utils")

app = modal.App("flux", image=flux_image)

SECRETS = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
    "AWS_S3_BUCKET": "eikona-io-ml",
    "HF_TOKEN": os.environ.get("HF_TOKEN"),
})

# Imports inside the image
with flux_image.imports():
    import torch
    from diffusers import FluxPipeline
    from diffusers.schedulers import (
        EulerDiscreteScheduler,  # https://huggingface.co/docs/diffusers/api/schedulers/euler
        DPMSolverMultistepScheduler,  # https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver
        DDIMScheduler,  # https://huggingface.co/docs/diffusers/api/schedulers/ddim
    )
    from utils.optimizer import optimize
    import boto3
    from utils.utils import _chunk_even


@app.function(image=flux_image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from schemas.schemas import GenerateRequest
    from utils.sampling_schedualer import make_sampling_plan
    from utils.utils import _chunk_even
    import asyncio, time
    from typing import List

    web_app = FastAPI()

    MODEL_POOL: List[FluxModel] = []
    POOL_LOCK = asyncio.Lock()

    async def _ensure_pool(n: int) -> List[FluxModel]:
        """Grow the pool to at least n instances; never shrink. Warm up concurrently."""
        if n <= 0:
            return []
        # 1) create missing stubs under lock
        async with POOL_LOCK:
            missing = n - len(MODEL_POOL)
            new_instances: List[FluxModel] = []
            if missing > 0:
                start = len(MODEL_POOL)
                new_instances = [
                    FluxModel(compile=IS_COMPILE_MODEL, id=start + i)
                    for i in range(missing)
                ]
                MODEL_POOL.extend(new_instances)
                print(f"[pool] grew to {len(MODEL_POOL)} (added {missing})")

        # 2) warm up all new instances concurrently (outside the lock)
        if new_instances:
            await asyncio.gather(
                *[inst.warmup.remote.aio() for inst in new_instances],
                return_exceptions=True,
            )

        return MODEL_POOL[:n]

    @web_app.post("/generate")
    async def generate(req: GenerateRequest):
        t0 = time.perf_counter()
        print(f"Received request, current pool size: {len(MODEL_POOL)}")

        # 1) one sampling dict per requested image
        plan = make_sampling_plan(req.nof_images, req.diversity)

        # 2) workers = requested gpus (capped by #jobs)
        workers = max(1, int(req.gpus or 1))
        workers = min(workers, len(plan))

        # 3) ensure we have at least `workers` FluxModel stubs
        instances = await _ensure_pool(workers)

        # 4) split plan into `workers` batches
        batches = _chunk_even(plan, workers)  # List[List[dict]]

        # 5) fire all batch jobs concurrently on chosen instances
        tasks = [
            inst.inference_batch.remote.aio(req.prompt, batch)
            for inst, batch in zip(instances, batches)
        ]
        batch_results = await asyncio.gather(*tasks)  # List[List[dict]]

        # 6) flatten
        images = [b for batch in batch_results for b in batch]
        latency_ms = int((time.perf_counter() - t0) * 1000)

        return JSONResponse(
            status_code=200,
            content={
                "prompt": req.prompt,
                "count": len(images), 
                "images": images,
                "latency_ms": latency_ms,
            },
        )

    return web_app


############################################################################################
@app.cls(
    gpu=f"L40S:1",
    scaledown_window=20 * 60,  # Scale down after 20 minutes of idleness
    timeout=60 * 60,  # 60 Minutes - leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
    secrets=[SECRETS],
)
class FluxModel:
    compile: bool = modal.parameter(default=False)
    id: int = modal.parameter(default=0)

    @modal.enter()
    def enter(self):
        pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe = optimize(pipe, compile=self.compile)
        self.bucket = os.environ.get("AWS_S3_BUCKET")
        self.region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
        self.s3 = boto3.client("s3", region_name=self.region)

    def _inference_once(
        self, prompt: str, sampling_params: dict, save_to_s3: bool
    ) -> bytes:
        # Set the sampling params
        width, height = sampling_params.get("width", 512), sampling_params.get(
            "height", 512
        )
        # scheduler = sampling_params.get("sampler", EulerDiscreteScheduler) # FIXME
        # self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config) #FIXME
        nof_steps = sampling_params.get("nof_steps", 20)
        cfg_scale = sampling_params.get("cfg_scale", 5.0)

        print(f"{self.id} 🎨 generating image...")
        out = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=cfg_scale,
            num_inference_steps=nof_steps,
            output_type="pil",
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        byte_stream.seek(0)  # <-- rewind to start
        length = byte_stream.getbuffer().nbytes
        if save_to_s3:
            filename=f"images/flux-{int(time.time()*1000)}.jpg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=filename,
                Body=byte_stream,
                ContentType="image/jpeg",
                ContentLength=length,
                ACL="public-read"
            )
            return {
                "url": f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{filename}",
                "sampling_params": sampling_params
            }

    @modal.method()
    def inference(self, prompt: str, sampling_params: dict) -> bytes:
        return self._inference_once(prompt, sampling_params, save_to_s3=True)

    @modal.method()
    def inference_batch(
        self, prompt: str, sampling_params_list: list[dict]
    ) -> list[bytes]:
        # Run the whole batch on ONE GPU/container
        return [
            self._inference_once(prompt, sp, save_to_s3=True)
            for sp in sampling_params_list
        ]

    @modal.method()
    def warmup(self) -> bool:
        # very small, fast run to load weights & compile kernels
        sp = {"width": 64, "height": 64, "nof_steps": 1, "cfg_scale": 1.0}
        _ = self._inference_once("warmup", sp, save_to_s3=False)
        return True
