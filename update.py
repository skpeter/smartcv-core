"""Self-update from GitHub latest release.zip / source.zip.

Stdlib-first. Optional certifi when present (frozen OpenSSL often has no CA).
Runs before paddle bootstrap. Git clones and DEV builds skip.
"""
from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_GITHUB_OWNER = "skpeter"
_PRESERVE = frozenset({"config.ini"})
_CHUNK = 256 * 1024
_UA = "SmartCV-updater"


def ensure_ca_bundle() -> None:
    """Point OpenSSL/requests at certifi CA when system/frozen store is empty."""
    if os.environ.get("SSL_CERT_FILE") and os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    try:
        import certifi

        ca = certifi.where()
    except Exception:
        return
    if not ca or not os.path.isfile(ca):
        return
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)


def maybe_update() -> None:
    try:
        ensure_ca_bundle()
        _maybe_update()
    except Exception as e:
        print(f"Update failed: {e}")
        print("Continuing on current build.")


def _maybe_update() -> None:
    local, repo = _read_build_info()
    if local is None or repo is None:
        return
    install = _install_root()
    if (install / ".git").exists():
        return

    remote = _fetch_remote_version(repo)
    if remote is None or remote <= local:
        return

    msg = f"New build {remote} available (you are on build {local}). Update now? [y/N] "
    if not sys.stdin.isatty():
        print(msg.strip())
        print("No TTY; skip apply.")
        return
    try:
        answer = input(msg).strip().lower()
    except EOFError:
        return
    if answer not in {"y", "yes"}:
        return

    asset = "release.zip" if getattr(sys, "frozen", False) else "source.zip"
    url = _asset_url(repo, asset)
    print(f"Downloading {asset}...")
    tmp = Path(tempfile.mkdtemp(prefix="smartcv-update-"))
    try:
        zpath = tmp / asset
        _download(url, zpath)
        extracted = tmp / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extracted)
        payload = _payload_root(extracted)
        if getattr(sys, "frozen", False):
            _apply_frozen(payload, install)
        else:
            _apply_source(payload, install)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _read_build_info():
    version = repo = None
    for name in ("build_info", "core.build_info"):
        try:
            mod = __import__(name, fromlist=["__version__", "repo"])
            version = getattr(mod, "__version__", None)
            repo = getattr(mod, "repo", None)
            break
        except Exception:
            continue
    if version is None:
        return None, None
    try:
        local = int(str(version).strip())
    except (TypeError, ValueError):
        return None, None
    if not repo:
        return None, None
    return local, str(repo).strip()


def _install_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _asset_url(repo: str, name: str) -> str:
    return f"https://github.com/{_GITHUB_OWNER}/{repo}/releases/latest/download/{name}"


def _ssl_context() -> ssl.SSLContext:
    ensure_ca_bundle()
    ca = os.environ.get("SSL_CERT_FILE")
    if ca and os.path.isfile(ca):
        return ssl.create_default_context(cafile=ca)
    return ssl.create_default_context()


def _urlopen(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def _fetch_remote_version(repo: str):
    url = _asset_url(repo, "version.txt")
    try:
        with _urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace").strip()
        return int(text)
    except Exception as e:
        # Broken CA / offline must never abort startup.
        print(f"Update check skipped: {e}")
        return None


def _download(url: str, dest: Path) -> None:
    with _urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = min(100, done * 100 // total)
                    print(f"\rDownloading... {pct}%", end="", flush=True)
                else:
                    print(f"\rDownloading... {done // (1024 * 1024)} MB", end="", flush=True)
    print()


def _payload_root(extracted: Path) -> Path:
    entries = [p for p in extracted.iterdir() if p.name != "__MACOSX"]
    exe_here = extracted / "smartcv.exe"
    if len(entries) == 1 and entries[0].is_dir() and not exe_here.exists():
        return entries[0]
    return extracted


def _overlay(src: Path, dest: Path) -> None:
    src = src.resolve()
    dest = dest.resolve()
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if rel.as_posix() in _PRESERVE or (len(rel.parts) == 1 and rel.name in _PRESERVE):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _apply_source(payload: Path, install: Path) -> None:
    print("Applying source update...")
    _overlay(payload, install)
    req = install / "core" / "requirements.txt"
    if req.is_file():
        print("Updating Python dependencies...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req)]
            )
        except subprocess.CalledProcessError as e:
            print(f"pip install failed: {e}")
    print("Restarting...")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _apply_frozen(payload: Path, install: Path) -> None:
    exe = payload / "smartcv.exe"
    if not exe.is_file():
        raise FileNotFoundError("release.zip missing smartcv.exe")
    dest_exe = install / "smartcv.exe"
    helper = Path(tempfile.gettempdir()) / "smartcv_apply_update.ps1"
    helper.write_text(_FROZEN_HELPER_PS1, encoding="utf-8")
    flags = 0
    if sys.platform == "win32":
        flags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    print("Restarting to finish update...")
    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(helper),
            str(os.getpid()),
            str(payload),
            str(install),
            str(dest_exe),
        ],
        close_fds=False,
        creationflags=flags,
    )
    sys.exit(0)


_FROZEN_HELPER_PS1 = r"""param(
  [Parameter(Mandatory=$true)][int]$WaitPid,
  [Parameter(Mandatory=$true)][string]$Src,
  [Parameter(Mandatory=$true)][string]$Dest,
  [Parameter(Mandatory=$true)][string]$Exe
)
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
  Start-Sleep -Milliseconds 400
}
Start-Sleep -Milliseconds 500
$srcFull = (Resolve-Path -LiteralPath $Src).Path
Get-ChildItem -LiteralPath $Src -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($srcFull.Length).TrimStart('\','/')
  if ($rel -eq 'config.ini') { return }
  $target = Join-Path $Dest $rel
  $dir = Split-Path $target
  if (-not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
  Copy-Item -LiteralPath $_.FullName -Destination $target -Force
}
Start-Process -FilePath $Exe -WorkingDirectory $Dest
$extractRoot = Split-Path $srcFull
if ((Split-Path $extractRoot -Leaf) -like 'smartcv-update-*') {
  Remove-Item -LiteralPath $extractRoot -Recurse -Force -ErrorAction SilentlyContinue
}
"""
