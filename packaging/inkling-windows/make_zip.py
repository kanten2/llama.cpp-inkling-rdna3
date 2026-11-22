"""Assemble a portable gfx1100 runtime from an extracted official b12606 release and the ROCm 10 SDK."""
import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = "inkling-rdna3-windows-gfx1100"
OFFICIAL = (
    "llama-server.exe", "llama-server-impl.dll", "llama-bench.exe", "llama-bench-impl.dll",
    "llama-common.dll", "mtmd.dll", "ggml.dll", "ggml-base.dll", "libomp.dll",
    "amdhip64_7.dll", "amd_comgr.dll", "rocm_kpack.dll",
)
BLAS = ("hipblas.dll", "rocblas.dll", "rocsolver.dll", "libhipblaslt.dll", "origami.dll")
# HIP and CLR distribute the same license text.
NOTICES = {
    "HIP-CLR.txt": ("ROCm/rocm-systems/97d8e3931860830a2ad634750299476f7e20a6cf/projects/hip/LICENSE.md",
                    "b185aaa652b0bf066c37a0d6314ce4bf4521e4a3c9bf46edd2f6a777ac522223"),
    "rocm-kpack.txt": ("ROCm/rocm-kpack/4ee2886d4b9000bd07af8a2cc73819d2c95077a7/LICENSE",
                       "f7ea04a2a919c9da80baed49c7bc27cc181b2c7fddee76ad82fd698a4f66ecbe"),
}


def package(args):
    files = {"chat.cmd": HERE/"chat.cmd", "bench.cmd": HERE/"bench.cmd", "tools/launch.ps1": HERE/"launch.ps1",
             "tools/messages.json": HERE/"messages.json",
             "README.md": HERE/"README.md", "licenses/llama.cpp.txt": ROOT/"LICENSE",
             "licenses/jsonhpp.txt": ROOT/"licenses/LICENSE-jsonhpp",
             "licenses/LLVM-OpenMP.txt": args.official/"LICENSE-LLVM-OpenMP"}
    for language in ("ja", "zh-Hans", "zh-Hant"):
        guide = f"README.{language}.md"
        files[guide] = HERE/guide
    for folder in (ROOT/"vendor").rglob("LICENSE*"):
        if folder.is_file():
            files["licenses/vendor/"+folder.relative_to(ROOT/"vendor").as_posix()] = folder
    for name in OFFICIAL:
        files["runtime/bin/"+name] = args.official/name
    cpus = sorted(args.official.glob("ggml-cpu-*.dll"))
    if not cpus:
        raise ValueError("Missing official CPU backends")
    for path in cpus:
        files["runtime/bin/"+path.name] = path
    for name in ("ggml-hip.dll", "llama.dll"):
        files["runtime/bin/"+name] = args.build_bin/name
    libs = args.sdk/"_rocm_sdk_libraries"
    for name in BLAS:
        files["runtime/bin/"+name] = libs/"bin"/name
    files["runtime/.kpack/blas_lib_gfx1100.kpack"] = libs/".kpack/blas_lib_gfx1100.kpack"
    for subdir in ("rocblas/library", "hipblaslt/library/gfx1100"):
        folder = libs/"bin"/subdir
        if not folder.is_dir() or not any(folder.iterdir()):
            raise ValueError(f"Missing ROCm kernel library: {subdir}")
        for path in folder.rglob("*"):
            if path.is_file():
                files["runtime/"+path.relative_to(libs).as_posix()] = path
    for component in ("amd_comgr", "half", "hipblas", "hipblas-common", "hipblaslt", "rocblas", "rocsolver", "origami"):
        notice, = (args.development_sdk/"share/doc"/component).glob("LICENSE*")
        files["licenses/"+component+notice.suffix] = notice

    assets = {}
    for name, (resource, digest) in NOTICES.items():
        with urlopen("https://raw.githubusercontent.com/"+resource, timeout=30) as resp:
            content = resp.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Unexpected notice digest: "+name)
        assets["licenses/"+name] = content

    expected = args.output
    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        raise FileExistsError(expected)
    with NamedTemporaryFile(dir=expected.parent, suffix=".partial", delete=False) as temp:
        tmp = Path(temp.name)
    checks = []
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for key in sorted(files.keys() | assets.keys()):
                data = files[key].read_bytes() if key in files else assets[key]
                if key.endswith(".cmd"):
                    data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                checks.append(f"{hashlib.sha256(data).hexdigest()}  {key}\n")
                z.writestr(NAME+"/"+key, data)
            z.writestr(NAME+"/SHA256SUMS.txt", "".join(checks))
        tmp.rename(expected)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{expected.name}  {expected.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("official", "build-bin", "sdk", "development-sdk", "output"):
        parser.add_argument("--"+key, required=True, type=Path)
    package(parser.parse_args())
