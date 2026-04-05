"""
Shared schedule utilities for non-linear interpolation across timesteps.

Supported schedule types: linear, quadratic, sqrt, cosine, exponential.
"""

import math

SCHEDULE_TYPES = ["linear", "quadratic", "sqrt", "cosine", "exponential"]


def warp_alpha(alpha, schedule_type="linear"):
    """
    Warp a linear progress value alpha ∈ [0, 1] using the given schedule type.

    All schedule types map 0 → 0 and 1 → 1, but differ in the intermediate curve.
    """
    if schedule_type == "linear":
        return alpha
    elif schedule_type == "quadratic":
        return alpha ** 2
    elif schedule_type == "sqrt":
        return alpha ** 0.5
    elif schedule_type == "cosine":
        return (1 - math.cos(alpha * math.pi)) / 2
    elif schedule_type == "exponential":
        return (math.exp(alpha) - 1) / (math.e - 1)
    else:
        raise ValueError(f"Unknown schedule type '{schedule_type}'. Choose from: {SCHEDULE_TYPES}")


def make_schedule(start, end, num_steps, schedule_type="linear"):
    """
    Generate a list of `num_steps` values from `start` to `end` using the given schedule type.

    For 'linear' this is equivalent to np.linspace(start, end, num_steps).tolist().
    """
    if num_steps == 1:
        return [start]
    values = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        warped = warp_alpha(alpha, schedule_type)
        values.append(start + (end - start) * warped)
    return values
