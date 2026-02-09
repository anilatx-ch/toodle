"""Check CUDA runtime readiness for TensorFlow in the active environment."""

from __future__ import annotations

import ctypes
import subprocess
import sys


def _check_shared_lib(name: str) -> tuple[bool, str]:
    try:
        ctypes.CDLL(name)
        return True, "ok"
    except OSError as exc:
        return False, str(exc)


def _run_nvidia_smi() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, proc.stdout.splitlines()[0] if proc.stdout else "ok"
    return False, (proc.stderr or proc.stdout or "nvidia-smi failed").strip()


def main() -> int:
    import tensorflow as tf

    print(f"TensorFlow: {tf.__version__}")
    build = tf.sysconfig.get_build_info()
    print(f"TF build CUDA: {build.get('cuda_version', 'unknown')}")
    print(f"TF build cuDNN: {build.get('cudnn_version', 'unknown')}")

    checks = [
        ("libcuda.so.1", "driver"),
        ("libcudnn.so.9", "cuDNN"),
        ("libnccl.so.2", "NCCL"),
    ]

    all_ok = True
    for lib, label in checks:
        ok, info = _check_shared_lib(lib)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {lib} ({info})")
        all_ok = all_ok and ok

    smi_ok, smi_info = _run_nvidia_smi()
    print(f"[{'PASS' if smi_ok else 'FAIL'}] nvidia-smi: {smi_info}")
    all_ok = all_ok and smi_ok

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected GPUs: {gpus}")
    if not gpus:
        print("[FAIL] TensorFlow cannot register any GPU device.")
        all_ok = False
    else:
        print("[PASS] TensorFlow GPU registration successful.")

    if not all_ok:
        print("CUDA check failed.")
        return 1

    print("CUDA check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
