"""Probe GPU. No admin. Missing/unreadable tools = no GPU."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class GpuInfo:
    kind: str  # nvidia | amd | apple | none
    name: str
    driver: tuple[int, ...] | None
    compute: float | None
    reason: str


def detect_gpu() -> GpuInfo:
    if sys.platform == "darwin":
        return GpuInfo("apple", "Apple", None, None, "macOS: MPS/CPU wheel")
    if sys.platform == "win32":
        info = _nvidia_smi()
        if info is not None:
            return info
        info = _windows_display_devices()
        if info is not None:
            return info
        return GpuInfo("none", "", None, None, "no NVIDIA GPU found")
    info = _nvidia_smi()
    if info is not None:
        return info
    info = _linux_rocm()
    if info is not None:
        return info
    return GpuInfo("none", "", None, None, "no NVIDIA/ROCm GPU found")


def _parse_driver(text: str) -> tuple[int, ...] | None:
    parts = []
    for bit in text.strip().split("."):
        if not bit.isdigit():
            break
        parts.append(int(bit))
    return tuple(parts) if parts else None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _nvidia_smi_cmds() -> list[list[str]]:
    bins = ["nvidia-smi"]
    if sys.platform == "win32":
        bins.extend([
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                         "System32", "nvidia-smi.exe"),
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ])
    query = [
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    return [[b, *query] for b in bins]


def _nvidia_smi() -> GpuInfo | None:
    for cmd in _nvidia_smi_cmds():
        result = _run(cmd)
        if result is None or result.returncode != 0:
            continue
        best: GpuInfo | None = None
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            name = parts[0]
            driver = _parse_driver(parts[1])
            compute = None
            if len(parts) >= 3:
                try:
                    compute = float(parts[2])
                except ValueError:
                    compute = None
            cand = GpuInfo("nvidia", name, driver, compute, f"nvidia-smi: {name}")
            if best is None or (compute or 0) > (best.compute or 0):
                best = cand
        if best is not None:
            return best
    return None


def _windows_display_devices() -> GpuInfo | None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("DeviceName", wintypes.WCHAR * 32),
            ("DeviceString", wintypes.WCHAR * 128),
            ("StateFlags", wintypes.DWORD),
            ("DeviceID", wintypes.WCHAR * 128),
            ("DeviceKey", wintypes.WCHAR * 128),
        ]

    EnumDisplayDevices = ctypes.windll.user32.EnumDisplayDevicesW
    EnumDisplayDevices.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.POINTER(DISPLAY_DEVICEW), wintypes.DWORD,
    ]
    EnumDisplayDevices.restype = wintypes.BOOL
    DISPLAY_DEVICE_ACTIVE = 0x00000001

    nvidia = None
    i = 0
    try:
        while True:
            dd = DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not EnumDisplayDevices(None, i, ctypes.byref(dd), 0):
                break
            i += 1
            name = dd.DeviceString or ""
            if dd.StateFlags & DISPLAY_DEVICE_ACTIVE and "nvidia" in name.lower():
                nvidia = name
                break
    except OSError:
        return None
    if nvidia:
        return GpuInfo(
            "nvidia", nvidia, None, None,
            f"display device: {nvidia} (no compute cap; conservative CUDA)",
        )
    return GpuInfo("none", "", None, None, "no NVIDIA display device")


def _linux_rocm() -> GpuInfo | None:
    kfd = "/dev/kfd"
    if not os.path.exists(kfd) or not os.access(kfd, os.R_OK):
        return None
    return GpuInfo("amd", "AMD ROCm", None, None, "ROCm /dev/kfd")
