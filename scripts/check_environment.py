"""Check runtime dependencies and required local controller model files."""

from __future__ import annotations

import hashlib
import importlib
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = PROJECT_ROOT / "final_model2.0" / "handheld_rtmdet_tiny_960.py"
MODEL_CHECKPOINT = PROJECT_ROOT / "final_model2.0" / "best_model.pth"
EXPECTED_CHECKPOINT_SHA256 = (
    "9efffdbb7ea60572a61af21da47688dce9d319f7e848435e61476fccecb4015f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    failures = []
    modules = ("numpy", "cv2", "torch", "mmengine", "mmcv", "mmdet", "mmpose")
    loaded = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            loaded[name] = module
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {name}: {version}")
        except Exception as error:  # dependency diagnostics should report all failures
            failures.append(f"{name}: {error}")
            print(f"[FAIL] {name}: {error}")

    try:
        from mmcv.ops import nms  # noqa: F401

        print("[OK] mmcv.ops.nms")
    except Exception as error:
        failures.append(f"mmcv.ops.nms: {error}")
        print(f"[FAIL] mmcv.ops.nms: {error}")

    torch = loaded.get("torch")
    if torch is not None:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print(f"FFmpeg: {shutil.which('ffmpeg') or 'not found (OpenCV fallback available)'}")

    for path in (MODEL_CONFIG, MODEL_CHECKPOINT):
        if path.is_file():
            print(f"[OK] {path.relative_to(PROJECT_ROOT)}")
        else:
            failures.append(f"missing file: {path}")
            print(f"[FAIL] missing file: {path}")

    if MODEL_CHECKPOINT.is_file():
        actual = sha256(MODEL_CHECKPOINT)
        if actual == EXPECTED_CHECKPOINT_SHA256:
            print("[OK] controller checkpoint SHA-256")
        else:
            failures.append("controller checkpoint SHA-256 mismatch")
            print(f"[FAIL] controller checkpoint SHA-256: {actual}")

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
