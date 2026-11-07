"""Light notebook bootstrap. Importing this module never installs anything."""
import importlib
import importlib.util
import subprocess
import sys


NOTEBOOK_DEPENDENCIES = {
    "filelock": "filelock>=3.13,<4",
    "numpy": "numpy>=1.26,<3",
    "pandas": "pandas>=2.1,<4",
    "PIL": "Pillow>=10,<14",
    "yaml": "PyYAML>=6,<7",
}


def prepare_notebook(install_missing=True):
    """Install missing lightweight imports into the kernel's own environment.

    Called explicitly by the notebook's first cell, before package imports.
    Does not install Torch/CUDA, load images, allocate a GPU or start experiments.
    """
    print(f"Python del kernel: {sys.executable}", flush=True)
    if sys.platform == "win32":
        print(
            "Kernel local de Windows. Para usar la T4: Seleccionar kernel > "
            "Colab > servidor T4. La instalación de dependencias no cambia el kernel.",
            flush=True,
        )
    else:
        print("Kernel Linux/macOS; la calibración comprobará la GPU CUDA disponible.", flush=True)
    missing = [package for name, package in NOTEBOOK_DEPENDENCIES.items()
               if importlib.util.find_spec(name) is None]
    if missing:
        if not install_missing:
            raise RuntimeError(
                "Faltan dependencias del kernel: " + ", ".join(missing)
                + ". Active INSTALL_MISSING_NOTEBOOK_DEPS en la primera celda."
            )
        print("Instalando dependencias básicas ausentes:", ", ".join(missing), flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "No se pudieron instalar dependencias en este kernel. Revise la salida "
                "de pip y la conexión; también puede instalar requirements.txt "
                "en el entorno seleccionado."
            ) from error
        importlib.invalidate_caches()
    else:
        print("Dependencias básicas del notebook disponibles.", flush=True)
