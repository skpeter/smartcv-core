# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

block_cipher = None

a = Analysis(
    ['../core/core.py'],
    pathex=['.', '../core', 'core'],
    binaries=collect_dynamic_libs('torch') + collect_dynamic_libs('cv2'),
    datas=collect_data_files('easyocr') + collect_data_files('torch') + collect_data_files('cv2'),
    hiddenimports=collect_submodules('easyocr') + [
        'numpy._core._exceptions',
        'scipy._cyutility',
        'scipy._lib.messagestream',
        'scipy._lib._ccallback_c',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.spatial.transform._rotation_groups',
        'scipy.spatial.transform._rotation',
        'scipy.special._ufuncs_cxx',
        'scipy.special._ufuncs',
        'scipy.special._specfun',
        'scipy.linalg._fblas',
        'scipy.linalg._flapack',
        'scipy.linalg._cythonized_array_utils',
        'cv2',
        'torch',
        'torchvision',
        'torchaudio',
        'easyocr',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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
    upx_exclude=['torch.dll', 'torch_global_deps.dll', 'python3.dll', '_uuid.pyd'],
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
    upx_exclude=[],
    name='smartcv'
)