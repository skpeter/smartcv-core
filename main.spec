# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Torch is excluded from the freeze and loaded at runtime from AppData.
# PyInstaller therefore never sees torch's import graph and omits unused
# stdlib modules (first crash: timeit via torch._strobelight). Pack stdlib
# so runtime torch/easyocr imports do not die on ModuleNotFoundError.
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


# Torchvision is also runtime-loaded (excluded). Its `from PIL import ImageDraw`
# never reaches the freeze graph. core.py only pulls Image/ImageFile, so the
# bundle had PIL C-exts + those two modules and died on ImageDraw.
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')


a = Analysis(
    ['../core/core.py'],
    pathex=['.', '../core', 'core'],
    binaries=pil_binaries,
    datas=pil_datas,
    hiddenimports=[
        'numpy._core._exceptions', 'scipy._cyutility',
        'packaging', 'packaging.utils', 'packaging.requirements',
        'packaging.markers', 'packaging.version',
        'gpu_detect', 'torch_bootstrap',
        'timeit',
        'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageColor',
        'PIL.ImageEnhance', 'PIL.ImageOps', 'PIL.ImageFilter',
    ] + pil_hiddenimports + _stdlib_hiddenimports(),

    hookspath=[os.path.join(SPECPATH, 'hooks')],
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'nvidia'],
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
    upx_exclude=['torch.dll', 'torch_global_deps.dll', 'python3.dll', '_uuid.pyd', 'c10.dll'],
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
    upx_exclude=['torch.dll', 'torch_global_deps.dll', 'python3.dll', '_uuid.pyd', 'c10.dll'],
    name='smartcv'
)