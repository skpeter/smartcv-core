# -*- mode: python ; coding: utf-8 -*-
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

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


def _collect_all(name):
    try:
        return collect_all(name)
    except Exception:
        try:
            datas = collect_data_files(name)
        except Exception:
            datas = []
        try:
            hidden = collect_submodules(name)
        except Exception:
            hidden = [name]
        return datas, [], hidden


def _copy_metadata(name):
    try:
        return copy_metadata(name, recursive=True)
    except TypeError:
        return copy_metadata(name)
    except Exception:
        return []


pil_datas, pil_binaries, pil_hiddenimports = _collect_all('PIL')
# Paddle imports these at startup, but paddle itself is excluded from analysis.
setuptools_datas, setuptools_binaries, setuptools_hiddenimports = _collect_all('setuptools')
pkg_resources_datas, pkg_resources_binaries, pkg_resources_hiddenimports = _collect_all('pkg_resources')
# Frozen OpenSSL has no system CA path; updater/paddle download need cacert.pem.
certifi_datas, certifi_binaries, certifi_hiddenimports = _collect_all('certifi')
paddleocr_datas, paddleocr_binaries, paddleocr_hiddenimports = _collect_all('paddleocr')
# paddleocr 3.x imports paddlex at module load; datas + dist-info must be in the freeze.
paddlex_datas, paddlex_binaries, paddlex_hiddenimports = _collect_all('paddlex')
paddlex_meta = _copy_metadata('paddlex')
paddleocr_meta = _copy_metadata('paddleocr')

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
    binaries=(pil_binaries + certifi_binaries + paddleocr_binaries + paddlex_binaries
              + setuptools_binaries + pkg_resources_binaries),
    datas=(pil_datas + certifi_datas + paddleocr_datas + paddlex_datas
           + paddlex_meta + paddleocr_meta
           + setuptools_datas + pkg_resources_datas),
    hiddenimports=[
        'numpy._core._exceptions', 'scipy._cyutility',
        'packaging', 'packaging.utils', 'packaging.requirements',
        'packaging.markers', 'packaging.version',
        'gpu_detect', 'paddle_bootstrap', 'update', 'ocr_parse',
        'certifi',
        'timeit',
        'setuptools.command.easy_install',
        'setuptools.command.build_ext',
        'setuptools.command.install',
        'setuptools.command.build',
        'distutils.command.build',
        'distutils.errors',
        'pkg_resources',
        'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageColor',
        'PIL.ImageEnhance', 'PIL.ImageOps', 'PIL.ImageFilter',
    ] + pil_hiddenimports + certifi_hiddenimports + paddleocr_hiddenimports
      + paddlex_hiddenimports + setuptools_hiddenimports
      + pkg_resources_hiddenimports + _stdlib_hiddenimports() + _scan_imports,

    hookspath=[],
    runtime_hooks=[],
    excludes=['paddle', 'paddlepaddle', 'paddlepaddle_gpu', 'nvidia'],
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
