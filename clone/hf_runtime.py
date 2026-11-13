try:
    import spaces

    _gpu = spaces.GPU
except ImportError:

    def _gpu(fn):
        return fn


@_gpu
def zerogpu_probe() -> None:
    """No-op. Exists only so ZeroGPU detects a @spaces.GPU function at startup."""
    return None
