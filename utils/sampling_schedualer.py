from itertools import product
from typing import Dict, Any, Iterable, List, Tuple
import random
import math
import os
from diffusers.schedulers import (
    EulerDiscreteScheduler,  # https://huggingface.co/docs/diffusers/api/schedulers/euler
    DPMSolverMultistepScheduler,  # https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver
    DDIMScheduler,  # https://huggingface.co/docs/diffusers/api/schedulers/ddim
)

def make_sampling_plan(
    num_samples: int = 1,
    diversity: Dict[str, Any] | None = None,
    *,
    base_seed: int | None = None,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build a list of sampling parameter dicts for generation.

    - num_samples: how many samples you intend to generate
    - diversity: dict like:
        {
          "samplers": ["euler", "ddim", "dpmpp_2m"],
          "nof_steps": [10, 20, 30],
          "cfg_scales": [3.5, 5.0, 7.5],
          "seeds": "fixed|random|mix"
        }
      Any field may be omitted; sensible defaults are used.
    - base_seed: used for deterministic shuffling and 'fixed'/'mix' modes.
    - shuffle: if True, combinations are shuffled (deterministically if base_seed is set).

    Returns: List[{"sampler": str, "steps": int, "cfg_scale": float, "seed": int}]
    """

    # --- defaults ---
    if diversity is None:
        diversity = {}
    samplers = diversity.samplers
    steps = diversity.nof_steps
    cfg_scales = diversity.cfg_scales
    seed_policy = diversity.seeds
    
    # map the str name of the sampler to the actual class (for later use)
    sampler_map = {
        "euler": EulerDiscreteScheduler.__name__,
        # "ddim": DDIMScheduler.__name__,  # Not supported in current model
        # "dpmpp_2m": DPMSolverMultistepScheduler.__name__,  # Not supported in current model
    }
    for s in samplers:
        if s not in sampler_map:
            print(f"⚠️  Warning: unknown sampler '{s}' (ignoring)")
    
    # map the sampler list to actual classes; ignore unknown names
    samplers = [sampler_map[s] for s in samplers if s in sampler_map]
    
    # full parameter space
    combos: List[Tuple[str, int, float]] = [(s, int(t), float(c)) for s, t, c in product(samplers, steps, cfg_scales)]

    # deterministic RNG if base_seed given; else system randomness
    rng = random.Random(base_seed if base_seed is not None else os.urandom(16))

    if shuffle:
        rng.shuffle(combos)

    M = len(combos)
    if M == 0:
        raise ValueError("No combinations available (empty samplers/steps/cfg_scales).")

    plan: List[Dict[str, Any]] = []

    # If we need N samples and have M unique combos:
    # - If N <= M: pick N evenly-spaced indices across the shuffled list (uniform coverage).
    # - If N  > M: cycle through the combos (repeat) until we have N.
    if num_samples <= M:
        # evenly spaced indices: floor(i * M / N), i=0..N-1 (unique when N <= M)
        indices = [ (i * M) // num_samples for i in range(num_samples) ]
    else:
        # cycle through all combos repeatedly
        indices = [ i % M for i in range(num_samples) ]

    for i, idx in enumerate(indices):
        sampler, nsteps, cfg = combos[idx]

        # --- seed selection policy ---
        if seed_policy == "fixed":
            seed = base_seed if base_seed is not None else 0
        elif seed_policy == "random":
            seed = rng.getrandbits(32)
        else:  # "mix": deterministic per (combo, i, base_seed) for diversity + reproducibility
            seed = (hash((sampler, nsteps, cfg, i, base_seed)) & 0xFFFFFFFF)

        plan.append({
            "sampler": sampler,
            "steps": int(nsteps),
            "cfg_scale": float(cfg),
            "seed": int(seed),
        })

    return plan

