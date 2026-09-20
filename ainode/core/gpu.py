"""GPU detection and capability reporting."""

from dataclasses import dataclass, field
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


@dataclass
class GPUDevice:
    """One NVIDIA device as the driver describes it.

    ``memory_total_mb`` is 0 when NVML will not report it, which is the
    unified-memory case (GB10 / DGX Spark: ``nvidia-smi`` prints ``[N/A]``).
    The pooled figure for such a node is on :class:`GPUSet`, counted once.
    """
    index: int
    name: str
    memory_total_mb: int = 0
    unified_memory: bool = False


@dataclass
class GPUSet:
    """Every NVIDIA device on this host, and the node's real memory total.

    ``detect_gpu`` answers for device 0 only, which reported a four-V100 host as
    one 32 GB GPU everywhere the number mattered: the announcement, the
    topology's GPU count and the cluster's total VRAM (#163). This type is the
    whole host: ``count`` devices and ``memory_total_mb`` summed across them.

    On a unified-memory part the pool is the host's RAM and belongs to every
    device at once, so it is counted ONCE rather than per device.
    """
    devices: list[GPUDevice] = field(default_factory=list)
    memory_total_mb: int = 0
    unified_memory: bool = False
    cuda_version: str = ""
    driver_version: str = ""
    compute_capability: str = ""

    @property
    def count(self) -> int:
        return len(self.devices)

    @property
    def name(self) -> str:
        """One name for the host: the device model when they all match."""
        names = {d.name for d in self.devices}
        if len(names) == 1:
            return next(iter(names))
        if not names:
            return "CPU"
        return " + ".join(sorted(names))

    @property
    def label(self) -> str:
        """Human form: ``4 x Tesla V100-SXM2-32GB``."""
        if self.count > 1:
            return f"{self.count} x {self.name}"
        return self.name


_gpu_cache: Optional[GPUInfo] = None
_gpu_set_cache: Optional[GPUSet] = None


def detect_gpu(use_cache: bool = True) -> Optional[GPUInfo]:
    """Detect NVIDIA GPU and return its capabilities.

    Device 0 only. Callers that need the node's real GPU count or memory total
    want :func:`detect_gpus`. See :class:`GPUSet`.
    """
    global _gpu_cache
    if use_cache and _gpu_cache is not None:
        return _gpu_cache
    try:
        import warnings
        # pynvml is deprecated upstream; nvidia-ml-py is the maintained fork
        # but has the same API. Suppress the warning — users shouldn't see it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Handle unified memory (DGX Spark / GB10). NVML either raises here or
        # returns a struct of zeros (the same reason `nvidia-smi` prints N/A),
        # so a zero total is the unified case too, not a 0 GB GPU.
        memory_total = memory_free = 0
        unified = False
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = int(mem.total) // (1024 * 1024)
            memory_free = int(mem.free) // (1024 * 1024)
        except Exception:
            memory_total = 0
        if not memory_total:
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


def detect_gpus(use_cache: bool = True) -> Optional[GPUSet]:
    """Enumerate EVERY NVIDIA device on this host and sum their memory.

    Returns None when there is no NVIDIA GPU (or no NVML to ask). A host whose
    driver will not report per-device memory is reported as unified memory with
    the host's RAM as the pool, counted once for the node rather than once per
    device.
    """
    global _gpu_set_cache
    if use_cache and _gpu_set_cache is not None:
        return _gpu_set_cache
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml
        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
        except Exception:
            count = 1
        devices: list[GPUDevice] = []
        for index in range(max(0, count)):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                continue
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            total_mb = 0
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = int(mem.total) // (1024 * 1024)
            except Exception:
                total_mb = 0
            devices.append(GPUDevice(
                index=index,
                name=str(name),
                memory_total_mb=total_mb,
                unified_memory=not total_mb,
            ))

        driver = ""
        cuda_str = ""
        compute = ""
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")
        except Exception:
            driver = ""
        try:
            raw_cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_str = f"{raw_cuda // 1000}.{(raw_cuda % 1000) // 10}"
        except Exception:
            cuda_str = ""
        if devices:
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(
                    pynvml.nvmlDeviceGetHandleByIndex(devices[0].index))
                compute = f"{major}.{minor}"
            except Exception:
                compute = ""
        pynvml.nvmlShutdown()

        if not devices:
            return None

        unified = any(d.unified_memory for d in devices)
        if unified:
            # One pool shared by every device AND by the host: count it once.
            import psutil
            total_mb = int(psutil.virtual_memory().total // (1024 * 1024))
        else:
            total_mb = sum(d.memory_total_mb for d in devices)

        result = GPUSet(
            devices=devices,
            memory_total_mb=total_mb,
            unified_memory=unified,
            cuda_version=cuda_str,
            driver_version=str(driver or ""),
            compute_capability=compute,
        )
        _gpu_set_cache = result
        return result
    except Exception:
        return None


def gpu_summary() -> str:
    """Return a human-readable GPU summary."""
    gpus = detect_gpus()
    if gpus is None:
        gpu = detect_gpu()
        if gpu is None:
            return "No NVIDIA GPU detected"
        mem_gb = gpu.memory_total_mb / 1024
        um = " (unified memory)" if gpu.unified_memory else ""
        return f"{gpu.name} | {mem_gb:.0f} GB{um} | CUDA {gpu.cuda_version} | SM {gpu.compute_capability}"

    mem_gb = gpus.memory_total_mb / 1024
    um = " (unified memory)" if gpus.unified_memory else ""
    return (f"{gpus.label} | {mem_gb:.0f} GB{um} | CUDA {gpus.cuda_version} "
            f"| SM {gpus.compute_capability}")
