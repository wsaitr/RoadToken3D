import torch
import math
from collections.abc import Mapping
from torch.nn import functional as F
DEFAULT_WEIGHTS = {
    "xy": 1.0,
    "height": 1.0,
    "route_set_classification": 1.0,
    "expected_xy_risk": 0.5,
    "graph_road_nll": 1.0,
    "arrival_upper_bound": 0.2,
}

def validate_loss_weights(overrides=None):
    """Small explicit positive-weight contract, separate from weight search."""
    if overrides is None:
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(overrides, Mapping) or set(overrides) - set(DEFAULT_WEIGHTS):
        raise ValueError("Only the six named residual-candidate loss weights are supported")
    result = dict(DEFAULT_WEIGHTS)
    for name, value in overrides.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0.0 < value <= 10.0):
            raise ValueError("All candidate loss weights must be finite and in (0, 10]")
        result[name] = float(value)
    return result

def _ratio(numerator, denominator):
    # Preserve the real denominator, including fractional confidence masses.
    # The 1 is solely an empty-set numerical guard; it is never a fake count.
    safe = torch.where(denominator > 0., denominator, torch.ones_like(denominator))
    return numerator / safe

def _masked_logsumexp(value, mask):
    """Exact masked log mass with a finite empty-set computational branch."""
    active = mask.any(-1)
    masked = value.masked_fill(~mask, float("-inf"))
    safe = torch.where(active[..., None], masked, torch.zeros_like(masked))
    result = torch.logsumexp(safe, -1)
    return torch.where(active, result, torch.zeros_like(result))

def _finite_on(value, mask, name):
    if not torch.isfinite(torch.where(mask, value, torch.zeros_like(value))).all():
        raise ValueError(f"Non-finite {name} at an active supervised position")
