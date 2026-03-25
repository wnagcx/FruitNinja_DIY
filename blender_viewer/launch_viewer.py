#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLEND = Path(__file__).resolve().with_name("viewer.blend")
DEFAULT_PLY = PROJECT_ROOT / "outputs" / "gs_fill.ply"
DEFAULT_BLENDER = Path("/mnt/d/blender/blender.exe")


def wsl_to_windows(path: Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def latest_epoch_ply(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("orange_demo_epoch_*.ply"))
    if not candidates:
        if DEFAULT_PLY.exists():
            print(
                f"No orange_demo_epoch_*.ply found in {outputs_dir}, "
                f"falling back to {DEFAULT_PLY}"
            )
            return DEFAULT_PLY
        raise FileNotFoundError(
            f"No orange_demo_epoch_*.ply found in {outputs_dir}"
        )
    return candidates[-1]


def resolve_watch_path(args: argparse.Namespace) -> Path:
    if args.watch_path is not None:
        watch_path = Path(args.watch_path).expanduser().resolve()
    elif args.latest_epoch:
        watch_path = latest_epoch_ply(PROJECT_ROOT / "outputs").resolve()
    else:
        watch_path = DEFAULT_PLY.resolve()

    if not watch_path.exists():
        raise FileNotFoundError(f"Watch path does not exist: {watch_path}")
    return watch_path


def build_command(args: argparse.Namespace, watch_path: Path) -> list[str]:
    blender_exe = Path(args.blender_exe).expanduser()
    if not blender_exe.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_exe}")

    runtime_script = Path(__file__).resolve().with_name("viewer_runtime.py")
    blend_path = Path(args.blend_path).expanduser().resolve()

    command = [str(blender_exe)]
    if blend_path.exists():
        command.append(wsl_to_windows(blend_path))

    command.extend(
        [
            "--python",
            wsl_to_windows(runtime_script),
            "--",
            "--watch-path",
            wsl_to_windows(watch_path),
            "--poll-seconds",
            str(args.poll_seconds),
            "--blend-path",
            wsl_to_windows(blend_path),
        ]
    )

    if args.auto_save_blend:
        command.append("--auto-save-blend")

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Blender FruitNinja viewer from WSL."
    )
    parser.add_argument(
        "--blender-exe",
        default=str(DEFAULT_BLENDER),
        help="Path to blender.exe from the WSL view.",
    )
    parser.add_argument(
        "--watch-path",
        default=None,
        help="PLY file to watch. Defaults to outputs/gs_fill.ply.",
    )
    parser.add_argument(
        "--latest-epoch",
        action="store_true",
        help="Watch the latest orange_demo_epoch_*.ply instead of gs_fill.ply.",
    )
    parser.add_argument(
        "--blend-path",
        default=str(DEFAULT_BLEND),
        help="Blend file to open or create.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval used by the Blender runtime script.",
    )
    parser.add_argument(
        "--auto-save-blend",
        action="store_true",
        help="Save viewer.blend after first scene setup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    watch_path = resolve_watch_path(args)
    command = build_command(args, watch_path)

    print("Launching Blender viewer with:")
    print(f"  watch_path = {watch_path}")
    print(f"  blend_path = {Path(args.blend_path).expanduser().resolve()}")
    print(f"  blender    = {Path(args.blender_exe).expanduser()}")

    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
