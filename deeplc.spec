# -*- mode: python ; coding: utf-8 -*-

import importlib.metadata

import PyInstaller.utils.hooks

# Collect all dependencies recursively
requirements = {req.split()[0] for req in importlib.metadata.requires("deeplc")}
requirements |= {
    "deeplc", "sklearn", "sklearn.utils", "sklearn.neighbors", "sklearn.tree",
    "distributed", "nicegui", "plotly", "pywebview", "onnx2torch",
}

hidden_imports = set()
datas = []
checked = set()
while requirements:
    requirement = requirements.pop()
    checked.add(requirement)
    if requirement in {"pywin32"}:
        continue
    try:
        importlib.metadata.version(requirement)
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError, ImportError):
        continue
    try:
        datas_, _, hidden_imports_ = PyInstaller.utils.hooks.collect_all(
            requirement, include_py_files=True
        )
    except ImportError:
        continue
    datas += datas_
    hidden_imports_ = set(hidden_imports_)
    hidden_imports_.discard("")
    hidden_imports_.discard(None)
    requirements |= hidden_imports_ - checked
    hidden_imports |= hidden_imports_

hidden_imports = sorted(
    h for h in hidden_imports if "tests" not in h.split(".") and "__pycache__" not in h
)
datas = [
    d for d in datas
    if "__pycache__" not in d[0] and d[1] not in {".", "build", "dist", "Output"}
]

a = Analysis(
    ["deeplc/__main__.py"],
    datas=datas,
    hiddenimports=hidden_imports,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="deeplc",
    console=False,
    icon="img/deeplc.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="deeplc",
)
