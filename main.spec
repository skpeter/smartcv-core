# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# PaddlePaddle is excluded from the freeze and loaded at runtime from AppData.
# Pack stdlib so runtime paddle/paddleocr imports do not die on ModuleNotFoundError.
_STDLIB_SKIP = {
    'site-packages', 'ensurepip', 'venv', 'turtledemo', 'idlelib',
    'test', 'tkinter', 'pydoc_data', 'distutils', '__pycache__',
    'lib2to3', 'config',
}


def _stdlib_hiddenimports():
    stdlib = Path(sysconfig.get_path('stdlib'))
    builtin = set(sys.builtin_module_names)
    names = []
    for p in sorted(stdlib.glob('*.py')):
        if p.stem in builtin or p.stem.startswith('__'):
            continue
        names.append(p.stem)
    for p in sorted(stdlib.iterdir()):
        if not p.is_dir() or p.name in _STDLIB_SKIP or p.name.startswith('.'):
            continue
        if p.name in builtin:
            continue
        names.append(p.name)
        try:
            names.extend(collect_submodules(p.name))
        except Exception:
            pass
    return names


pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')
# Frozen OpenSSL has no system CA path; updater/paddle download need cacert.pem.
certifi_datas, certifi_binaries, certifi_hiddenimports = collect_all('certifi')
paddleocr_datas, paddleocr_binaries, paddleocr_hiddenimports = collect_all('paddleocr')

_scan_file = Path(SPECPATH) / 'paddle_hiddenimports.txt'
_scan_imports = []
if _scan_file.exists():
    _scan_imports = [
        ln.strip() for ln in _scan_file.read_text(encoding='utf-8').splitlines()
        if ln.strip() and not ln.startswith('#')
    ]


a = Analysis(
    ['../core/core.py'],
    pathex=['.', '../core', 'core'],
    binaries=pil_binaries + certifi_binaries + paddleocr_binaries,
    datas=pil_datas + certifi_datas + paddleocr_datas,
    hiddenimports=[
        'numpy._core._exceptions', 'scipy._cyutility',
        'packaging', 'packaging.utils', 'packaging.requirements',
        'packaging.markers', 'packaging.version',
        'gpu_detect', 'paddle_bootstrap', 'update',
        'certifi',
        'timeit',
        'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageColor',
        'PIL.ImageEnhance', 'PIL.ImageOps', 'PIL.ImageFilter',
    ] + pil_hiddenimports + certifi_hiddenimports + paddleocr_hiddenimports
      + _stdlib_hiddenimports() + _scan_imports,

    hookspath=[os.path.join(SPECPATH, 'hooks')],
    runtime_hooks=[],
    excludes=['paddle', 'paddlepaddle', 'paddlepaddle_gpu', 'torch',
              'torchvision', 'torchaudio', 'nvidia'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='smartcv',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['python3.dll', '_uuid.pyd'],
    runtime_tmpdir=None,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../core/icon.ico',
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['python3.dll', '_uuid.pyd'],
    name='smartcv'
)
