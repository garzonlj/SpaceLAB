# Build transport-only snapshots of the current code into experimental notebooks.
# This writes files; it never executes Python, imports the pipeline or starts a job.
param([string[]]$NotebookPaths)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$utf8 = New-Object System.Text.UTF8Encoding($false)
$files = @(
    Get-ChildItem (Join-Path $repoRoot 'src/spacelab') -File -Filter '*.py'
    Get-ChildItem (Join-Path $repoRoot 'configs') -File -Filter '*.yaml'
    Get-ChildItem (Join-Path $repoRoot 'experiments') -Recurse -File -Filter '*.yaml'
    Get-ChildItem (Join-Path $repoRoot 'kit_participantes') -File -Filter '*.py'
    Get-Item (Join-Path $repoRoot 'requirements.txt')
)
$memory = New-Object System.IO.MemoryStream
$archive = New-Object System.IO.Compression.ZipArchive($memory, [System.IO.Compression.ZipArchiveMode]::Create, $true)
try {
    foreach ($file in ($files | Sort-Object FullName)) {
        $relative = $file.FullName.Substring($repoRoot.Length + 1).Replace('\', '/')
        $entry = $archive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
        $entry.LastWriteTime = [DateTimeOffset]::new(2026, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
        $stream = $entry.Open()
        try {
            $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
            $stream.Write($bytes, 0, $bytes.Length)
        } finally { $stream.Dispose() }
    }
} finally { $archive.Dispose() }
$bundleBytes = $memory.ToArray()
$memory.Dispose()
$base64 = [Convert]::ToBase64String($bundleBytes)
$hasher = [System.Security.Cryptography.SHA256]::Create()
try { $sha = ([BitConverter]::ToString($hasher.ComputeHash($bundleBytes))).Replace('-', '').ToLowerInvariant() }
finally { $hasher.Dispose() }

$bootstrapTemplate = @'
from pathlib import Path
import base64
import hashlib
import io
import subprocess
import sys
import zipfile

# This cell carries code/configuration only. No images, weights or results.
# Editing src/configs locally requires rebuilding this snapshot with
# scripts/package_colab_notebooks.ps1 before running a remote notebook.
_CODE_BUNDLE_SHA256 = '__SHA__'
_CODE_BUNDLE = '__BUNDLE__'
REPO_ROOT = None  # Optional explicit checkout path; normally leave automatic.
INSTALL_MISSING_NOTEBOOK_DEPS = True
RUN_SETUP = False  # Full training dependencies; no experiments are launched.
RUN_MOUNT_DRIVE = False
DATASET_DRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t'
DATASET_ROOT = '/content/spacelab_dataset'
STORAGE = '/content/drive/MyDrive/SpaceLAB_experiments/session_01'

if REPO_ROOT is not None:
    REPO_ROOT = Path(REPO_ROOT).expanduser().resolve()
elif sys.platform != 'win32' and Path('/content').is_dir():
    # Version-specific directory: never overwrite an older checkout or dataset.
    REPO_ROOT = Path('/content/SpaceLAB_runtime') / _CODE_BUNDLE_SHA256[:16]
else:
    REPO_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                      if (p/'src/spacelab/notebook_setup.py').is_file()), None)
    if REPO_ROOT is None:
        raise FileNotFoundError('Abra el notebook dentro de SpaceLAB o conecte el kernel Colab.')

if sys.platform != 'win32' and Path('/content/SpaceLAB_runtime') == REPO_ROOT.parent:
    _payload = base64.b64decode(_CODE_BUNDLE)
    if hashlib.sha256(_payload).hexdigest() != _CODE_BUNDLE_SHA256:
        raise ValueError('El paquete de código está incompleto o alterado; recargue el notebook.')
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    _root = REPO_ROOT.resolve()
    with zipfile.ZipFile(io.BytesIO(_payload)) as _archive:
        for _entry in _archive.infolist():
            _target = (_root / _entry.filename).resolve()
            if _root not in _target.parents:
                raise ValueError('Ruta fuera del paquete de código: ' + _entry.filename)
            _contents = _archive.read(_entry)
            if _target.exists():
                if _target.read_bytes() != _contents:
                    raise RuntimeError('Código remoto editado en ' + str(_target)
                                       + '. Consérvelo y configure REPO_ROOT explícitamente.')
            else:
                _target.parent.mkdir(parents=True, exist_ok=True)
                _target.write_bytes(_contents)
    del _payload

if not (REPO_ROOT/'src/spacelab/notebook_setup.py').is_file():
    raise FileNotFoundError('REPO_ROOT no contiene el paquete actualizado: ' + str(REPO_ROOT))
print('Código del proyecto disponible en:', REPO_ROOT)
if RUN_MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
if RUN_SETUP:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(REPO_ROOT/'requirements.txt')])
sys.path.insert(0, str(REPO_ROOT/'src'))
# Reload the complete project package when selecting a different snapshot.
# Run the configuration and action cells again after this bootstrap.
import importlib
_loaded = sys.modules.get('spacelab')
if _loaded is not None and Path(_loaded.__file__).resolve().parent != (REPO_ROOT/'src/spacelab').resolve():
    for _module_name in list(sys.modules):
        if _module_name == 'spacelab' or _module_name.startswith('spacelab.'):
            del sys.modules[_module_name]
    importlib.invalidate_caches()
    print('SpaceLAB actualizado en este kernel. Continúe ejecutando las celdas en orden.', flush=True)
from spacelab.notebook_setup import prepare_notebook
prepare_notebook(install_missing=INSTALL_MISSING_NOTEBOOK_DEPS)
from spacelab.time_budget import launch
from spacelab.splits import load_protocol
from spacelab.experiment_registry import config_from, collect, freeze_selection
from spacelab.io import read_json
from IPython.display import display, Markdown
PROTOCOL_PATH = Path(STORAGE)/'protocol.json'
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_BENCHMARK = False
RUN_SPLITS = False
ENABLE_OPTIONAL = False
SELECTED_RUN = None
RUN_PATH = None
VALIDATION_RUN = None
RESUME_CHECKPOINT = None
'@
$bootstrap = $bootstrapTemplate.Replace('__SHA__', $sha).Replace('__BUNDLE__', $base64)
$updated = 0
if ($NotebookPaths) {
    $notebookFiles = @($NotebookPaths | ForEach-Object { Get-Item (Join-Path $repoRoot $_) })
} else {
    $notebookFiles = @(Get-ChildItem (Join-Path $repoRoot 'experiments') -Recurse -File -Filter '*.ipynb')
    if (Test-Path (Join-Path $repoRoot 'SpaceLAB_Colab.ipynb')) {
        $notebookFiles += Get-Item (Join-Path $repoRoot 'SpaceLAB_Colab.ipynb')
    }
}
foreach ($path in $notebookFiles) {
    $nb = [System.IO.File]::ReadAllText($path.FullName) | ConvertFrom-Json
    $changed = $false
    foreach ($cell in $nb.cells) {
        if ($cell.cell_type -ne 'code') { continue }
        if (($cell.source -join '').Contains('from spacelab.time_budget import launch')) {
            $cell.source = $bootstrap
            $cell.outputs = @()
            $cell.execution_count = $null
            $changed = $true
        }
    }
    if ($changed) {
        [System.IO.File]::WriteAllText($path.FullName, ($nb | ConvertTo-Json -Depth 40), $utf8)
        $updated++
    }
}
Write-Output "Notebooks con arranque autocontenido: $updated"
Write-Output "Paquete: $($files.Count) archivos de código/configuración, $($bundleBytes.Length) bytes comprimidos."
Write-Output "SHA-256: $sha"
