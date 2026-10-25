"""Same prefix operation with a padding-stable Ascend implementation.

NPU FP64 matrix accumulation is not assumed. This conservative on-device
FP32 sequential scan preserves exact zero-padded tails and autograd. It has
not been speed-qualified on NPU; no host tensor fallback is performed.
"""
import torch

def cumsum(value, dim, *, dtype=None):
    if value.device.type != 'npu':
        return torch.cumsum(value, dim, dtype=dtype)
    return scan(value, dim, dtype=dtype)

def scan(value, dim, *, dtype=None):
    """Explicit scan also exposed for small CPU implementation checks."""
    dim %= value.ndim
    result_dtype = dtype
    if result_dtype is None:
        result_dtype = torch.int64 if value.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32) else value.dtype
    work_dtype = torch.int32 if result_dtype == torch.int64 else result_dtype
    x = value.to(work_dtype)
    if x.shape[dim] == 0:
        return x.to(result_dtype)
    parts = x.unbind(dim)
    total = torch.zeros_like(parts[0])
    result = []
    for part in parts:
        total = total + part
        result.append(total)
    return torch.stack(result, dim=dim).to(result_dtype)
