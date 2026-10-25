"""Road-only XYZ learning. Accepted forecasts lie on supplied 3D polylines.

No Cartesian fallback or free height offset. Missing anchors cause explicit
abstention. Geometry membership is not certification of traffic legality or DEM.
"""
from .prefix import cumsum as _prefix_sum
import torch
from torch.nn import functional as F
from .route_geometry import NativeRouteGeometry

class StrictRoadGeometry(NativeRouteGeometry):

    def continuous_prefix(self, roads, directions, count, start_s, tolerance=0.001):
        (_, _, entry, exit, next_entry, _, _) = self.route_parts(roads, directions, count, start_s)
        gap = torch.linalg.vector_norm(next_entry - exit, dim=-1)
        bad_before = torch.cat((torch.zeros_like(gap[..., :1], dtype=torch.bool), gap[..., :-1] > tolerance), -1)
        real = torch.arange(roads.shape[-1], device=roads.device) < count[..., None]
        prefix = real & (_prefix_sum(bad_before.to(torch.int32), -1, dtype=torch.int32) == 0)
        return (prefix.long().sum(-1), gap)

def select_prediction(output):
    """Public output boundary: no candidate means NaN coordinates, zero mass."""
    valid = output['forecast_valid']
    choice = output['mode_logits'].argmax(-1)
    selected = output['trajectories'][torch.arange(len(choice), device=choice.device), choice]
    selected = torch.where(valid[:, None, None], selected, torch.full_like(selected, float('nan')))
    probability = F.softmax(output['mode_logits'], -1) * output['mode_valid']
    probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-08)
    return dict(prediction=selected, forecast_valid=valid, selected_mode=torch.where(valid, choice, torch.full_like(choice, -1)), probability=probability)
