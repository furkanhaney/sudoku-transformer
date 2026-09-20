#!/usr/bin/env python3
"""Focused tests for the repository-local CUDA toolkit materializer."""

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "setup_cuda", pathlib.Path(__file__).with_name("setup_cuda.py")
)
assert SPEC is not None and SPEC.loader is not None
setup_cuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_cuda)


class MaterializeCudaAliasesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name) / ".cuda"
        self.contents = {}
        for index, source_relative in enumerate(setup_cuda.CUDA_ALIASES.values()):
            if source_relative in self.contents:
                continue
            source = self.root / source_relative
            source.parent.mkdir(parents=True, exist_ok=True)
            content = f"library-{index}\n".encode()
            source.write_bytes(content)
            self.contents[source_relative] = content
        for alias_relative, source_relative in setup_cuda.CUDA_ALIASES.items():
            alias = self.root / alias_relative
            target = pathlib.Path(source_relative).name
            if alias.name.endswith((".13", ".10", ".4")):
                os.symlink(target, alias)
            else:
                soname = next(
                    pathlib.Path(other).name
                    for other, source in setup_cuda.CUDA_ALIASES.items()
                    if source == source_relative and other != alias_relative
                )
                os.symlink(soname, alias)

    def tearDown(self):
        self.temporary.cleanup()

    def test_materializes_each_chain_from_its_final_file(self):
        self.assertEqual(
            setup_cuda.CUDA_ALIASES,
            {
                "lib/libcudart.so": "lib/libcudart.so.13.2.51",
                "lib/libcudart.so.13": "lib/libcudart.so.13.2.51",
                "lib/libcurand.so": "lib/libcurand.so.10.4.2.51",
                "lib/libcurand.so.10": "lib/libcurand.so.10.4.2.51",
                "nvvm/lib64/libnvvm.so": "nvvm/lib64/libnvvm.so.4.0.0",
                "nvvm/lib64/libnvvm.so.4": "nvvm/lib64/libnvvm.so.4.0.0",
            },
        )
        setup_cuda.materialize_cuda_aliases(self.root)
        for alias_relative, source_relative in setup_cuda.CUDA_ALIASES.items():
            alias = self.root / alias_relative
            self.assertTrue(alias.is_file())
            self.assertFalse(alias.is_symlink())
            self.assertEqual(alias.read_bytes(), self.contents[source_relative])
            self.assertEqual(
                hashlib.sha256(alias.read_bytes()).digest(),
                hashlib.sha256((self.root / source_relative).read_bytes()).digest(),
            )

    def test_rerun_is_idempotent_and_repairs_corrupt_regular_alias(self):
        setup_cuda.materialize_cuda_aliases(self.root)
        corrupt = self.root / next(iter(setup_cuda.CUDA_ALIASES))
        corrupt.write_bytes(b"corrupt")
        setup_cuda.materialize_cuda_aliases(self.root)
        self.assertEqual(
            corrupt.read_bytes(),
            self.contents[setup_cuda.CUDA_ALIASES[str(corrupt.relative_to(self.root))]],
        )
        setup_cuda.materialize_cuda_aliases(self.root)

    def _replace_source(self, replacement):
        source_relative = next(iter(set(setup_cuda.CUDA_ALIASES.values())))
        source = self.root / source_relative
        source.unlink()
        replacement(source)

    def test_rejects_missing_source(self):
        self._replace_source(lambda _source: None)
        with self.assertRaisesRegex(RuntimeError, "required CUDA path is missing"):
            setup_cuda.materialize_cuda_aliases(self.root)

    def test_rejects_source_symlink_escaping_tree(self):
        outside = pathlib.Path(self.temporary.name) / "outside.so"
        outside.write_bytes(b"outside")
        self._replace_source(lambda source: os.symlink(outside, source))
        with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
            setup_cuda.materialize_cuda_aliases(self.root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable")
    def test_rejects_special_source(self):
        self._replace_source(os.mkfifo)
        with self.assertRaisesRegex(RuntimeError, "must be a regular file"):
            setup_cuda.materialize_cuda_aliases(self.root)

    def test_rejects_directory_source(self):
        self._replace_source(lambda source: source.mkdir())
        with self.assertRaisesRegex(RuntimeError, "must be a regular file"):
            setup_cuda.materialize_cuda_aliases(self.root)

    def test_rejects_unexpected_link(self):
        os.symlink("missing", self.root / "unexpected")
        with self.assertRaisesRegex(RuntimeError, "unexpected CUDA symlink.*unexpected"):
            setup_cuda.materialize_cuda_aliases(self.root)

    def test_main_materializes_aliases_when_every_package_marker_skips(self):
        package_names = (
            "cuda_cudart",
            "cuda_crt",
            "cuda_cccl",
            "cuda_nvcc",
            "cuda_tileiras",
            "libnvvm",
            "libcurand",
        )
        manifest = {}
        for index, name in enumerate(package_names):
            digest = f"{index:064x}"
            (self.root / f".{name}.sha256").write_text(digest + "\n")
            manifest[name] = {
                "linux-x86_64": {
                    "sha256": digest,
                    "size": 1,
                    "relative_path": f"unused-{name}.tar.xz",
                }
            }
        response = io.BytesIO(json.dumps(manifest).encode())
        with (
            mock.patch.object(setup_cuda, "DEST", self.root),
            mock.patch.object(setup_cuda.platform, "system", return_value="Linux"),
            mock.patch.object(setup_cuda.platform, "machine", return_value="x86_64"),
            mock.patch.object(setup_cuda.urllib.request, "urlopen", return_value=response) as urlopen,
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            setup_cuda.main()
        self.assertEqual(urlopen.call_count, 1)
        self.assertFalse(any(path.is_symlink() for path in self.root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
