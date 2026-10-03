"""Phase 0 environment check: Python, PyTorch, CUDA, GPU, and free VRAM."""

import platform
import sys


def main() -> int:
    print(f"Python   {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    try:
        import torch
    except ImportError:
        print("PyTorch  not installed")
        return 1

    print(f"PyTorch  {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA     not available")
        return 1

    print(f"CUDA     {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        print(f"GPU {i}    {name}: {free / 2**30:.1f} / {total / 2**30:.1f} GiB free")

    try:
        import lerobot

        print(f"LeRobot  {getattr(lerobot, '__version__', 'installed')}")
    except ImportError:
        print("LeRobot  not installed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
