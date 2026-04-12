"""GPU detection and capability reporting."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a detected NVIDIA GPU."""
    name: str
    memory_total_mb: int
    memory_free_mb: int
    cuda_version: str
    driver_version: str
    compute_capability: str
    unified_memory: bool = False  # DGX Spark / GB10


_gpu_cache: Optional[GPUInfo] = None


def detect_gpu(use_cache: bool = True) -> Optional[GPUInfo]:
    """Detect NVIDIA GPU and return its capabilities."""
    global _gpu_cache
    if use_cache and _gpu_cache is not None:
        return _gpu_cache
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Handle unified memory (DGX Spark / GB10)
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = mem.total // (1024 * 1024)
            memory_free = mem.free // (1024 * 1024)
            unified = False
        except pynvml.NVMLError:
            import psutil
            total = psutil.virtual_memory().total // (1024 * 1024)
            memory_total = total
            memory_free = total  # Approximate
            unified = True

        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")

        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        cuda_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

        pynvml.nvmlShutdown()

        result = GPUInfo(
            name=name,
            memory_total_mb=memory_total,
            memory_free_mb=memory_free,
            cuda_version=cuda_str,
            driver_version=driver,
            compute_capability=f"{major}.{minor}",
            unified_memory=unified,
        )
        _gpu_cache = result
        return result
    except Exception:
        return None


def gpu_summary() -> str:
    """Return a human-readable GPU summary."""
    gpu = detect_gpu()
    if gpu is None:
        return "No NVIDIA GPU detected"

    mem_gb = gpu.memory_total_mb / 1024
    um = " (unified memory)" if gpu.unified_memory else ""
    return f"{gpu.name} | {mem_gb:.0f} GB{um} | CUDA {gpu.cuda_version} | SM {gpu.compute_capability}"
