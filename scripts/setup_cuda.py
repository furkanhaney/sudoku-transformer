#!/usr/bin/env python3
"""Install a minimal CUDA 13.2 toolkit in this study's ignored .cuda directory.

Uses NVIDIA's redistributable manifest and verifies each archive's SHA-256.
Requires Python 3.12+ (tarfile's data extraction filter), Linux x86_64, and an
existing compatible NVIDIA driver. Does not install or change the driver.
"""

import hashlib
import io
import json
import pathlib
import platform
import tarfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEST = pathlib.Path(__file__).resolve().parents[1] / ".cuda"


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
    print(f"CUDA_TOOLKIT_PATH={DEST}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
