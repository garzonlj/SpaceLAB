# Embed the current preparation implementation in the recovery cell, so it can
# run with an older imported SpaceLAB package without restarting the kernel.
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$bytes = [System.IO.File]::ReadAllBytes((Join-Path $repoRoot 'src/spacelab/preparation.py'))
$payload = [Convert]::ToBase64String($bytes)
$hasher = [System.Security.Cryptography.SHA256]::Create()
try { $sha = ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-','').ToLowerInvariant() }
finally { $hasher.Dispose() }
$loader = @'
    # BEGIN PREPARATION COMPATIBILITY PATCH
    # This implementation travels with this cell, independently of old imports.
    from pathlib import Path
    import base64
    import hashlib
    import importlib.util
    import inspect
    import sys
    if sys.platform == 'win32' or not Path('/content').is_dir():
        raise RuntimeError('Seleccione el kernel Colab del mismo servidor T4.')
    _preparation_source = base64.b64decode('__PAYLOAD__')
    _preparation_sha = '__SHA__'
    if hashlib.sha256(_preparation_source).hexdigest() != _preparation_sha:
        raise RuntimeError('Parche de preparación incompleto; recargue el notebook desde disco.')
    _patch_directory = Path('/content/SpaceLAB_runtime_patches')/_preparation_sha[:16]
    _patch_directory.mkdir(parents=True, exist_ok=True)
    _patch_file = _patch_directory/'preparation.py'
    if _patch_file.exists():
        if _patch_file.read_bytes() != _preparation_source:
            raise RuntimeError('El parche remoto fue editado. No se sobrescribe automáticamente.')
    else:
        _patch_file.write_bytes(_preparation_source)
    _patch_name = 'spacelab._preparation_resume_'+_preparation_sha[:16]
    if _patch_name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_patch_name, _patch_file)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_patch_name] = _module
        try:
            _spec.loader.exec_module(_module)
        except BaseException:
            sys.modules.pop(_patch_name, None)
            raise
    _preparation = sys.modules[_patch_name]
    stage_dataset = _preparation.stage_dataset
    verify_layout = _preparation.verify_layout
    if 'resume' not in inspect.signature(stage_dataset).parameters:
        raise RuntimeError('No se cargó la preparación que admite reanudación.')
    print('Preparación con progreso y reanudación cargada en este kernel.', flush=True)
    # END PREPARATION COMPATIBILITY PATCH
'@
$loader = $loader.Replace('__PAYLOAD__',$payload).Replace('__SHA__',$sha)
$path = Join-Path $repoRoot 'experiments/00_validation_protocol/00_create_splits.ipynb'
$nb = [System.IO.File]::ReadAllText($path) | ConvertFrom-Json
$updated = $false
foreach ($cell in $nb.cells) {
    if ($cell.id -ne 'cell-5') { continue }
    $source = $cell.source -join ''
    if ($source.Contains('# BEGIN PREPARATION COMPATIBILITY PATCH')) {
        $source = [regex]::Replace($source,'(?ms)^    # BEGIN PREPARATION COMPATIBILITY PATCH.*?^    # END PREPARATION COMPATIBILITY PATCH[^\r\n]*', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $loader })
    } else {
        $needle = '    from spacelab.preparation import stage_dataset, verify_layout'
        if (-not $source.Contains($needle)) { throw 'No se encontró la importación de preparación.' }
        $source = $source.Replace($needle,$loader)
    }
    $cell.source = $source
    $updated = $true
}
if (-not $updated) { throw 'No se encontró la celda de preparación.' }
[System.IO.File]::WriteAllText($path,($nb|ConvertTo-Json -Depth 40),(New-Object System.Text.UTF8Encoding($false)))
Write-Output 'Celda final actualizada: carga su propia función compatible con resume=True, sin reiniciar el kernel.'
Write-Output "SHA-256 del módulo de preparación: $sha"
