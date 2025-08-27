from typing import Literal, Optional, List
from pydantic import BaseModel, Field, conint, confloat

# Optional diversity knobs with sensible defaults
class DiversityConfig(BaseModel):
    # allow any subset of samplers; default to a good spread
    samplers: List[str] = Field(
        default=["euler", "ddim", "dpmpp_2m"],
        description="Samplers to try"
    )
    nof_steps: List[conint(gt=0)] = Field(
        default=[10, 20, 30],
        description="Candidate step counts for sampling"
    )
    cfg_scales: List[confloat(gt=0)] = Field(
        default=[3.5, 5.0, 7.5],
        description="Classifier-free guidance scales"
    )
    seeds: Literal["fixed", "random", "mix"] = Field(
        default="mix",
        description="How to seed runs"
    )

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt")
    nof_images: conint(gt=0) = Field(16, description="Number of images to produce")
    gpus: conint(gt=0) = Field(1, description="How many GPUs to use")
    diversity: Optional[DiversityConfig] = Field(
        default=None,
        description="Optional diversity controls; defaults used if omitted"
    )
    stream: bool = Field(False, description="If true, stream links as they’re ready")
    top_k: Optional[conint(gt=0)] = Field(
        default=None,
        description="If set, return only best K after scoring"
    )

# (Optional) response schema stub—adjust to your real return type
class GenerateResponse(BaseModel):
    ok: bool
    received: GenerateRequest
    
    
# class ImageMeta(BaseModel):
#     sampler: str
#     steps: conint(gt=0)
#     seed: conint(ge=0)

# class GeneratedImage(BaseModel):
#     url: AnyHttpUrl
#     meta: ImageMeta

# class GenerateResponse(BaseModel):
#     prompt: str
#     count: conint(ge=0)
#     images: List[GeneratedImage] = Field(default_factory=list)
#     latency_ms: conint(ge=0)