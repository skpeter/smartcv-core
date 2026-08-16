"""First-run PyTorch install. Frozen exe has no torch; wheels go to AppData."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests

try:
    from .gpu_detect import GpuInfo, detect_gpu
except ImportError:
    from gpu_detect import GpuInfo, detect_gpu

TORCH_VERSION = "2.13.0"
TORCHVISION_VERSION = "0.28.0"
MARKER_NAME = "smartcv-torch.json"
# Bump when install policy changes so _sync can repair deps (not full torch).
BOOTSTRAP_REV = 2
LOCK_NAME = ".setup.lock"
STALE_LOCK_SEC = 2 * 60 * 60

# Do not unpack these into the vendor dir — frozen copies stay authoritative.
SKIP_DEPS = {
    "numpy", "pillow", "pil", "opencv-python", "opencv-contrib-python",
    "setuptools", "pip", "wheel", "easyocr", "requests",
}

PYPI_SIMPLE = "https://pypi.org/simple"


def ensure_torch() -> None:
    gpu = detect_gpu()
    variant = pick_variant(gpu)
    pkg_dir = _pkg_dir()
    py_tag = f"{sys.version_info.major}.{sys.version_info.minor}"

    if not getattr(sys, "frozen", False) and _site_torch_ok(variant):
        return

    _with_lock(pkg_dir.parent, lambda: _install_if_needed(
        pkg_dir, variant, py_tag, gpu,
    ))
    _activate(pkg_dir)
    _verify(variant)


def pick_variant(gpu: GpuInfo) -> str:
    if sys.platform == "darwin":
        return "cpu"
    if gpu.kind == "amd" and sys.platform.startswith("linux"):
        return "rocm72"
    if gpu.kind != "nvidia":
        return "cpu"
    driver = gpu.driver or (0,)
    compute = gpu.compute
    # CUDA 13 drops < sm_75. Unknown compute → cu126 (Maxwell–Hopper).
    if compute is not None and compute < 7.5:
        return "cu126" if driver >= (560,) else "cpu"
    if driver >= (580,):
        return "cu130"
    if driver >= (560,):
        return "cu126"
    if gpu.driver is None:
        return "cu126"
    return "cpu"


def _pkg_dir() -> Path:
    override = os.environ.get("SMARTCV_TORCH_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "SmartCV" / "torch"


def _site_torch_ok(wanted: str) -> bool:
    """Source/dev: reuse existing torch if it covers the wanted backend."""
    import importlib.util
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return False
    version_py = Path(spec.origin).resolve().parent / "version.py"
    try:
        text = version_py.read_text(encoding="utf-8")
    except OSError:
        return False
    cuda = None
    for line in text.splitlines():
        if line.startswith("cuda") and "=" in line:
            rhs = line.split("=", 1)[1].strip()
            if rhs in ("None", "none"):
                cuda = None
            else:
                cuda = rhs.strip("\"'")
            break
    if wanted == "cpu":
        return True
    if wanted.startswith("cu"):
        return bool(cuda)
    if wanted.startswith("rocm"):
        return "rocm" in text.lower() or "hip" in text.lower()
    return True


def _read_marker(pkg_dir: Path) -> dict | None:
    path = pkg_dir / MARKER_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_marker(pkg_dir: Path, variant: str, py_tag: str) -> None:
    data = {
        "variant": variant,
        "torch": TORCH_VERSION,
        "py": py_tag,
        "platform": sys.platform,
        "rev": BOOTSTRAP_REV,
    }
    (pkg_dir / MARKER_NAME).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _marker_ok(pkg_dir: Path, variant: str, py_tag: str) -> bool:
    m = _read_marker(pkg_dir)
    if not m:
        return False
    if m.get("py") != py_tag or m.get("platform") != sys.platform:
        return False
    if m.get("torch") != TORCH_VERSION:
        return False
    have = m.get("variant")
    if have == variant:
        return True
    # CUDA torch can run CPU; skip re-download if forcing CPU.
    if variant == "cpu" and have and have.startswith("cu"):
        return True
    return False


def _with_lock(parent: Path, fn) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / LOCK_NAME
    if lock.exists():
        try:
            age = time.time() - lock.stat().st_mtime
            if age > STALE_LOCK_SEC:
                shutil.rmtree(lock, ignore_errors=True)
        except OSError:
            pass
    deadline = time.time() + 60 * 30
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.time() > deadline:
                raise RuntimeError("Timed out waiting for PyTorch setup lock")
            time.sleep(0.5)
    try:
        fn()
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def _install_if_needed(
    pkg_dir: Path, variant: str, py_tag: str, gpu: GpuInfo,
) -> None:
    if _marker_ok(pkg_dir, variant, py_tag) and (pkg_dir / "torch").is_dir():
        _install_wheels(pkg_dir, variant, repair=True)
        _write_marker(pkg_dir, variant, py_tag)
        return
    print("SmartCV first-time setup: installing PyTorch (once, ~0.2–3 GB).")
    print(f"GPU: {gpu.reason}")
    print(f"Selected: torch {TORCH_VERSION} ({variant})")
    staging = pkg_dir.parent / "torch.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _install_wheels(staging, variant)
        _write_marker(staging, variant, py_tag)
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        staging.rename(pkg_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("PyTorch setup done.")


def _index_for(variant: str) -> str:
    plat = sys.platform
    if variant == "cpu":
        if plat == "win32" or plat == "darwin":
            return PYPI_SIMPLE
        return "https://download.pytorch.org/whl/cpu"
    if variant == "cu126":
        return "https://download.pytorch.org/whl/cu126"
    if variant == "cu130":
        if plat.startswith("linux"):
            return PYPI_SIMPLE
        return "https://download.pytorch.org/whl/cu130"
    if variant == "rocm72":
        return "https://download.pytorch.org/whl/rocm7.2"
    return PYPI_SIMPLE


def _wheel_tags() -> tuple[str, str]:
    impl = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if sys.platform == "win32":
        plat = "win_amd64"
    elif sys.platform == "darwin":
        plat = "arm64" if os.uname().machine == "arm64" else "x86_64"
    else:
        machine = os.uname().machine
        plat = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    return impl, plat


def _compat(filename: str, impl: str, plat: str) -> bool:
    from packaging.utils import parse_wheel_filename
    name = unquote(filename.split("?")[0].split("/")[-1])
    if not name.endswith(".whl"):
        return False
    try:
        _n, _v, _b, tags = parse_wheel_filename(name)
    except Exception:
        return False
    for tag in tags:
        interp_ok = (
            tag.interpreter in (impl, "py3", "py2.py3")
            or tag.interpreter.replace("cp", "") == impl.replace("cp", "")
        )
        abi_ok = tag.abi in ("none", "abi3", impl)
        plat_ok = plat in tag.platform or tag.platform == "any"
        if interp_ok and abi_ok and plat_ok:
            return True
    return False


def _pep503_name(project: str) -> str:
    return re.sub(r"[-_.]+", "-", project).lower()


def _list_files(index_url: str, project: str) -> list[dict]:
    url = f"{index_url.rstrip('/')}/{_pep503_name(project)}/"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        raise FileNotFoundError(f"No index page for {project} at {url}")
    r.raise_for_status()
    files = []
    for m in re.finditer(r'href=["\']([^"\']+)["\']', r.text, re.I):
        href = m.group(1)
        full = urljoin(url, href).split("#")[0]
        fname = unquote(full.split("?")[0].split("/")[-1])
        if fname.endswith(".whl"):
            files.append({"filename": fname, "url": full.split("?")[0], "yanked": False})
    if not files:
        raise FileNotFoundError(f"No wheels listed for {project} at {url}")
    return files


def _pick_from_index(
    index_url: str,
    project: str,
    version: str | None,
    specifier: str | None = None,
) -> tuple[str, str]:
    from packaging.specifiers import SpecifierSet
    from packaging.utils import parse_wheel_filename
    impl, plat = _wheel_tags()
    files = _list_files(index_url, project)
    specset = SpecifierSet(specifier) if specifier else None
    idx = index_url.lower()
    require = None
    if "/cu" in idx:
        require = "+cu"
    elif "rocm" in idx:
        require = "+rocm"
    elif "/cpu" in idx:
        require = "+cpu"
    best = None
    best_ver = None
    for f in files:
        if f.get("yanked") or not f.get("url"):
            continue
        fname = f["filename"]
        if require and require not in fname:
            continue
        if not _compat(fname, impl, plat):
            continue
        try:
            _n, ver, _b, _t = parse_wheel_filename(
                unquote(fname.split("?")[0].split("/")[-1])
            )
        except Exception:
            continue
        if ver.is_prerelease:
            continue
        if version and ver.base_version != version:
            continue
        if specset is not None and ver not in specset:
            continue
        if best_ver is None or ver > best_ver:
            best_ver = ver
            best = f
    if best is None:
        raise FileNotFoundError(
            f"No wheel for {project} {version or specifier or ''} ({impl}, {plat}) at {index_url}"
        )
    return best["url"], best["filename"]


def _pick_wheel(
    index_url: str,
    project: str,
    version: str | None,
    specifier: str | None = None,
) -> tuple[str, str]:
    try:
        return _pick_from_index(index_url, project, version, specifier)
    except (FileNotFoundError, requests.HTTPError):
        # Never substitute PyPI torch — that is CPU (or a different CUDA).
        if _pep503_name(project) in ("torch", "torchvision", "torchaudio"):
            raise
        if index_url.rstrip("/") != PYPI_SIMPLE:
            return _pick_from_index(PYPI_SIMPLE, project, version, specifier)
        raise


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        last_print = 0
        with open(dest, "wb") as out:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                out.write(chunk)
                done += len(chunk)
                if total and done - last_print > 32 * 1024 * 1024:
                    last_print = done
                    print(f"  {done / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GB")
    print(f"  saved {dest.name}")


def _extract_wheel(whl: Path, target: Path) -> str:
    """Extract wheel; return dist-info METADATA text."""
    metadata = ""
    with zipfile.ZipFile(whl) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/") or ".." in Path(name).parts:
                continue
            out_name = name
            marker = ".data/purelib/"
            if marker in name:
                out_name = name.split(marker, 1)[1]
            elif ".data/" in name:
                continue
            dest = target / out_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if name.endswith(".dist-info/METADATA"):
                metadata = (target / out_name).read_text(encoding="utf-8", errors="replace")
    return metadata


def _requires(metadata: str) -> list[tuple[str, str | None]]:
    from packaging.requirements import Requirement
    out: list[tuple[str, str | None]] = []
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.marker and not req.marker.evaluate():
            continue
        name = req.name.lower().replace("_", "-")
        if name in SKIP_DEPS:
            continue
        spec = str(req.specifier) if req.specifier else None
        out.append((req.name, spec))
    return out


def _dists(pkg_dir: Path) -> dict[str, tuple[Path, str]]:
    """canonical name -> (dist-info dir, version string)."""
    found: dict[str, tuple[Path, str]] = {}
    for meta in pkg_dir.glob("*.dist-info/METADATA"):
        name = version = ""
        try:
            text = meta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
            if name and version:
                break
        if name and version:
            found[_pep503_name(name)] = (meta.parent, version)
    return found


def _project_ok(
    pkg_dir: Path,
    project: str,
    version: str | None,
    specifier: str | None,
) -> tuple[bool, str]:
    """Return (ok, metadata_text). ok means installed version satisfies pin/spec."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    dists = _dists(pkg_dir)
    hit = dists.get(_pep503_name(project))
    if not hit:
        return False, ""
    dist_info, ver_s = hit
    meta_path = dist_info / "METADATA"
    try:
        meta = meta_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, ""
    try:
        ver = Version(ver_s)
    except Exception:
        return False, meta
    if ver.is_prerelease:
        return False, meta
    if version and ver.base_version != version:
        return False, meta
    if specifier:
        try:
            if ver not in SpecifierSet(specifier):
                return False, meta
        except Exception:
            return False, meta
    return True, meta


def _wipe_project(pkg_dir: Path, project: str) -> None:
    key = _pep503_name(project)
    dists = _dists(pkg_dir)
    hit = dists.get(key)
    if hit:
        shutil.rmtree(hit[0], ignore_errors=True)
    for child in list(pkg_dir.iterdir()):
        if not child.is_dir():
            continue
        if _pep503_name(child.name) == key:
            shutil.rmtree(child, ignore_errors=True)


def _install_wheels(target: Path, variant: str, repair: bool = False) -> None:
    index = _index_for(variant)
    downloads = target / "_wheels"
    downloads.mkdir(exist_ok=True)
    seen = {n.lower().replace("_", "-") for n in SKIP_DEPS}
    queue: list[tuple[str, str | None, str | None]] = [
        ("torch", TORCH_VERSION, None),
        ("torchvision", TORCHVISION_VERSION, None),
    ]
    while queue:
        project, version, specifier = queue.pop(0)
        key = project.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        ok, meta = _project_ok(target, project, version, specifier)
        if ok:
            for dep, spec in _requires(meta):
                dkey = dep.lower().replace("_", "-")
                if dkey not in seen:
                    queue.append((dep, None, spec))
            continue
        if repair:
            print(f"Repairing {project} ({specifier or version or 'latest'})")
        url, fname = _pick_wheel(index, project, version, specifier)
        whl = downloads / unquote(fname.split("?")[0].split("/")[-1])
        _wipe_project(target, project)
        _download(url, whl)
        meta = _extract_wheel(whl, target)
        for dep, spec in _requires(meta):
            dkey = dep.lower().replace("_", "-")
            if dkey not in seen:
                queue.append((dep, None, spec))
    shutil.rmtree(downloads, ignore_errors=True)


def _activate(pkg_dir: Path) -> None:
    path = str(pkg_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        added = set()
        patterns = [
            "torch/lib",
            "nvidia/*/bin",
            "nvidia/*/lib",
            "nvidia/*/lib/x64",
        ]
        for pat in patterns:
            for folder in pkg_dir.glob(pat):
                if folder.is_dir() and folder not in added:
                    os.add_dll_directory(str(folder))
                    added.add(folder)
        os.environ["PATH"] = os.pathsep.join(
            [str(p) for p in added] + [os.environ.get("PATH", "")]
        )


def _verify(variant: str) -> None:
    import torch
    import torchvision  # noqa: F401
    import sympy  # noqa: F401
    cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    print(f"PyTorch {torch.__version__}  CUDA available: {cuda}")
    if variant.startswith("cu") and not cuda:
        print("CUDA wheel loaded but GPU not usable. OCR will use CPU.")
