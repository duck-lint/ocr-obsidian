from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal


OverwriteMode = Literal["never", "if_same_run", "always"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class OverwriteError(RuntimeError):
    pass


class MissingPathError(RuntimeError):
    pass


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def natural_sort_paths(paths: Iterable[Path]) -> list[Path]:
    token_pattern = re.compile(r"(\d+)")

    def key_for(path: Path) -> list[object]:
        parts = token_pattern.split(path.name.lower())
        keyed: list[object] = []
        for part in parts:
            if part.isdigit():
                keyed.append(int(part))
            else:
                keyed.append(part)
        return keyed

    return sorted(paths, key=key_for)


def discover_images(scans_path: Path) -> list[Path]:
    if not scans_path.exists():
        raise MissingPathError(f"Scan path does not exist: {scans_path}")
    if not scans_path.is_dir():
        raise MissingPathError(f"Scan path is not a directory: {scans_path}")
    image_paths = [p for p in scans_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    if not image_paths:
        raise MissingPathError(f"No supported image files found in: {scans_path}")
    return natural_sort_paths(image_paths)


def is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_parent(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _remove_existing(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def check_write_allowed(
    path: Path,
    overwrite: OverwriteMode,
    dry_run: bool,
    run_root: Path | None = None,
) -> None:
    if not path.exists():
        return
    if overwrite == "always":
        if not dry_run:
            _remove_existing(path)
        return
    if overwrite == "if_same_run":
        if run_root is not None and is_subpath(path, run_root):
            if not dry_run:
                _remove_existing(path)
            return
        raise OverwriteError(
            f"Refusing to overwrite outside the active run with --overwrite if_same_run: {path}"
        )
    raise OverwriteError(f"Refusing to overwrite existing path with --overwrite never: {path}")


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
