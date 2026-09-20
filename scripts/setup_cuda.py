#!/usr/bin/env python3
"""Install a minimal CUDA 13.2 toolkit in this study's ignored .cuda directory.

Uses NVIDIA's redistributable manifest and verifies each archive's SHA-256.
Requires Python 3.12+ (tarfile's data extraction filter), Linux x86_64, and an
existing compatible NVIDIA driver. Does not install or change the driver.
"""

import hashlib
import io
import json
import os
import pathlib
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEST = pathlib.Path(__file__).resolve().parents[1] / ".cuda"

CUDA_ALIASES = {
    "lib/libcudart.so": "lib/libcudart.so.13.2.51",
    "lib/libcudart.so.13": "lib/libcudart.so.13.2.51",
    "lib/libcurand.so": "lib/libcurand.so.10.4.2.51",
    "lib/libcurand.so.10": "lib/libcurand.so.10.4.2.51",
    "nvvm/lib64/libnvvm.so": "nvvm/lib64/libnvvm.so.4.0.0",
    "nvvm/lib64/libnvvm.so.4": "nvvm/lib64/libnvvm.so.4.0.0",
}


def _require_plain_path(root: pathlib.Path, relative: str, *, regular: bool) -> pathlib.Path:
    """Return an in-tree path whose existing components contain no links."""
    path = root / relative
    current = root
    for part in pathlib.PurePosixPath(relative).parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise RuntimeError(f"required CUDA path is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"CUDA path must not be a symlink: {relative}")
        if current != path and not stat.S_ISDIR(mode):
            raise RuntimeError(f"CUDA path parent is not a directory: {relative}")
    mode = path.lstat().st_mode
    expected = stat.S_ISREG(mode) if regular else stat.S_ISDIR(mode)
    if not expected:
        kind = "regular file" if regular else "directory"
        raise RuntimeError(f"CUDA path must be a {kind}: {relative}")
    return path


def materialize_cuda_aliases(root: pathlib.Path) -> None:
    """Replace the toolkit's fixed shared-library aliases with durable copies."""
    root = pathlib.Path(root)
    _require_plain_path(root.parent, root.name, regular=False)
    sources = {
        relative: _require_plain_path(root, relative, regular=True)
        for relative in set(CUDA_ALIASES.values())
    }
    allowed_links = {root / relative for relative in CUDA_ALIASES}
    unexpected = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_symlink() and path not in allowed_links
    )
    if unexpected:
        raise RuntimeError("unexpected CUDA symlink(s): " + ", ".join(unexpected))

    for alias_relative, source_relative in CUDA_ALIASES.items():
        source = sources[source_relative]
        alias = root / alias_relative
        _require_plain_path(root, str(alias.parent.relative_to(root)), regular=False)
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=f".{alias.name}.", dir=alias.parent)
            os.close(descriptor)
            temporary = pathlib.Path(name)
            shutil.copy2(source, temporary, follow_symlinks=False)
            with temporary.open("r+b") as copied:
                copied.flush()
                os.fsync(copied.fileno())
            os.replace(temporary, alias)
            temporary = None
            directory = os.open(alias.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    residual = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()
    )
    if residual:
        raise RuntimeError("residual CUDA symlink(s): " + ", ".join(residual))


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This helper supports Linux x86_64 only; use a system CUDA toolkit.")
    with urllib.request.urlopen(BASE + "redistrib_13.2.0.json", timeout=60) as response:
        manifest = json.load(response)
    DEST.mkdir(exist_ok=True)
    for name in ("cuda_cudart", "cuda_crt", "cuda_cccl", "cuda_nvcc", "cuda_tileiras", "libnvvm", "libcurand"):
        package = manifest[name]["linux-x86_64"]
        marker = DEST / ("." + name + ".sha256")
        if marker.is_file() and marker.read_text().strip() == package["sha256"]:
            print(f"Already installed: {name}", flush=True)
            continue
        print(f"Downloading {name}: {int(package['size']) / 1024**2:.2f} MiB", flush=True)
        with urllib.request.urlopen(BASE + package["relative_path"], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != package["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
            for member in archive.getmembers():
                parts = pathlib.PurePosixPath(member.name).parts
                if len(parts) > 1:
                    member.name = str(pathlib.PurePosixPath(*parts[1:]))
                    archive.extract(member, DEST, filter="data")
        marker.write_text(package["sha256"] + "\n")
    materialize_cuda_aliases(DEST)
    print(f"CUDA_TOOLKIT_PATH={DEST}")


if __name__ == "__main__":
    main()
