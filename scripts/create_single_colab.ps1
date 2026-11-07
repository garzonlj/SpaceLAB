# Only writes the web notebook; never executes Python or experimental code.
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$utf8 = New-Object System.Text.UTF8Encoding($false)
$template = [System.IO.File]::ReadAllText((Join-Path $repoRoot 'experiments/00_validation_protocol/00_prepare_data.ipynb')) | ConvertFrom-Json
$cells = New-Object System.Collections.Generic.List[object]
function Markdown([string]$s) { $cells.Add(@{cell_type='markdown';metadata=@{};source=$s;id=('cell-'+$cells.Count)}) }
function Code([string]$s) { $cells.Add(@{cell_type='code';metadata=@{};execution_count=$null;outputs=@();source=$s;id=('cell-'+$cells.Count)}) }
Markdown @'
# SpaceLAB — un solo Colab web con acceso a Drive

Sube **este único archivo** a https://colab.research.google.com/ con Archivo → Subir
notebook. Selecciona Entorno de ejecución → Cambiar tipo → GPU T4. Ejecuta las
celdas en orden y autoriza Google Drive cuando lo solicite.

El código está incluido: no requiere GitHub, VS Code ni subir carpetas de Python.
La preparación funciona con tu cuenta autorizada y esta
[carpeta compartida](https://drive.google.com/drive/folders/1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t).
No modifica los datos originales; resultados y checkpoints quedan en tu Drive.

Con los valores predeterminados se comprueba GPU y se prepara el dataset, pero
**no se entrena automáticamente**. Para iniciar los experimentos, activa
RUN_PIPELINE en la configuración. Todas las etapas se realizan en este notebook.
La preparación de datos y del entorno ocurre antes del reloj experimental.
Este notebook se entrega sin ejecutar ni inventar resultados o tiempos.
'@
Markdown '## 1. Preparar el código incluido'
Code ($template.cells[1].source -join '')
Markdown '## 2. Configuración editable'
Code @'
INSTALL_DEPENDENCIES = True
RUN_PREPARATION = True
RUN_PIPELINE = False  # Actívalo solo cuando estés listo para empezar los 90 minutos.
FREEZE_FINAL_SELECTION = False
RUN_FINAL_TEST = False
RUN_SUBMISSION = False
DATASET_FOLDER_ID = '1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t'
ZIP_FILE_ID = None  # Solo necesario si hay varios ZIP candidatos.
MOUNTED_SOURCE = '/content/drive/MyDrive/spacelab_dataset'  # Opcional; hay alternativa por API.
DATASET_ROOT = '/content/spacelab_dataset'
STORAGE = '/content/drive/MyDrive/SpaceLAB_experiments/session_01'
PROTOCOL_PATH = Path(STORAGE)/'protocol.json'
EXPERIMENT_IDS = ['M01','M02']
# Opcionales: R02 (896), A02 (geometría), A03 (fotometría), H02 (LR), etc.
ENABLE_OPTIONAL = False
GROUPS_CSV = None
SELECTED_RUN = None
SELECTED_INFERENCE_RUN = None
print('Resultados:', STORAGE, '| Entrenamiento activado:', RUN_PIPELINE)
'@
Markdown @'
## 3. Confirmar la GPU de Google y preparar dependencias

La salida debe mostrar la GPU real. Si no hay CUDA, seleccionar T4 antes de seguir.
Se conserva el par torch/torchvision de Colab. No se descargan pesos en esta celda.
'@
Code @'
import importlib
if sys.platform=='win32' or not Path('/content').is_dir():
    raise RuntimeError('Abre este notebook en Colab WEB y selecciona GPU T4.')
import torch
import torchvision
if not torch.cuda.is_available():
    raise RuntimeError('No hay GPU CUDA. Entorno de ejecución > Cambiar tipo > GPU T4.')
print('GPU de Google:',torch.cuda.get_device_name(0))
print('VRAM (GiB):',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
print('Torch:',torch.__version__,'| Torchvision:',torchvision.__version__,'| CUDA:',torch.version.cuda)
if INSTALL_DEPENDENCIES:
    packages = [s.strip() for s in (REPO_ROOT/'requirements.txt').read_text().splitlines()
                if s.strip() and not s.lstrip().startswith('#')
                and not s.strip().startswith(('torch>=','torchvision>='))]
    subprocess.check_call([sys.executable,'-m','pip','install',*packages])
    importlib.invalidate_caches()
print('Entorno preparado. La descarga de pesos solo ocurre al calibrar o entrenar.')
'@
Markdown @'
## 4. Autorizar Drive y preparar los datos

Acepta el diálogo de Google y usa la cuenta con acceso a la carpeta compartida.
Se intenta la ruta montada; si no existe, se usa la API autorizada con el ID del
enlace. Se prefiere un ZIP; una carpeta descomprimida puede tardar más al copiar
archivos. No se cargan todas las imágenes en RAM. Todavía no inicia el reloj.
'@
Code @'
if RUN_PREPARATION:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    from spacelab.drive_data import prepare_from_drive
    DATASET_ROOT = str(prepare_from_drive(DATASET_FOLDER_ID,DATASET_ROOT,
        mounted_source=MOUNTED_SOURCE,zip_file_id=ZIP_FILE_ID))
    Path(STORAGE).mkdir(parents=True,exist_ok=True)
    print('Dataset listo:',DATASET_ROOT)
    print('Resultados persistentes:',STORAGE)
else:
    print('Preparación desactivada.')
'@
Markdown '## 5. Control común, logs y presupuesto persistente'
Code @'
import time
import yaml
from spacelab.training import calibration_key
from spacelab.experiment_registry import config_from,collect,freeze_selection
from spacelab.splits import load_protocol
from spacelab.visualizations import comparison_figures
catalog = yaml.safe_load((REPO_ROOT/'configs/experiments.yaml').read_text(encoding='utf-8'))['experiments']
catalog = {r['id']:r for r in catalog}

def run_action(action,payload,seconds,final=False):
    print(f'Iniciando {action}. Techo: {seconds:.0f} s; espera la salida de esta celda.',flush=True)
    job = launch(action,payload,STORAGE,cap_seconds=seconds,final=final)
    print('Estado:',job['status'],'| Motivo:',job.get('reason'),flush=True)
    if job.get('run'):
        print('Archivos:',job['run'])
        log = Path(job['run'])/'worker.log'
        if log.exists():
            print(log.read_text(encoding='utf-8',errors='replace')[-5000:])
    return job

def budget_status():
    file = Path(STORAGE)/'budget.json'
    if not file.exists():
        print('El presupuesto todavía no ha empezado.')
        return
    state = read_json(file)
    elapsed = max(0,time.time()-state['started_at'])
    print(f'Tiempo desde inicio: {elapsed/60:.1f} min. Restante total: {max(0,5400-elapsed)/60:.1f} min.')
    print('Reserva final: 8 min; seguridad: 2 min. Las pausas también cuentan.')

def choose_architecture():
    table = collect(STORAGE)
    if table.empty or 'score' not in table:
        return None
    candidates = table[table.id.isin(['M01','M02']) & table.comparable & table.score.notna()]
    if candidates.empty or candidates.cohort.nunique()!=1:
        return None
    return candidates.sort_values(['score','map50'],ascending=False).iloc[0]['path']

budget_status()
'@
Markdown @'
## 6. Sesión experimental: splits → calibración → entrenamiento → validación

Solo se ejecuta con RUN_PIPELINE=True. Se reutiliza el mismo protocolo y STORAGE.
Primero se completa y valida M01; luego se intenta M02 y las variantes que hayas
seleccionado si su calibración y proyección caben. No se ejecutan experimentos en
paralelo. Test y oculto permanecen reservados. El total de calibración tiene techo
de cinco minutos y todas las acciones posteriores consumen los mismos 90 minutos.
'@
Code @'
if RUN_PIPELINE:
    from spacelab.preparation import verify_layout
    DATASET_ROOT = str(verify_layout(DATASET_ROOT))
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Primero montar Drive y preparar datos en la sección 4.')
    if not PROTOCOL_PATH.exists():
        job = run_action('splits',{'dataset':DATASET_ROOT,'groups_csv':GROUPS_CSV},300)
        if job['status']!='completado':
            raise RuntimeError('No se completaron los splits. Revisa el log antes de entrenar.')
    protocol = load_protocol(PROTOCOL_PATH,DATASET_ROOT)
    display(protocol['distributions'])
    for experiment_id in EXPERIMENT_IDS:
        candidate = catalog[experiment_id]
        if not candidate['enabled'] and not ENABLE_OPTIONAL:
            print(experiment_id,'omitido: opcional desactivado')
            continue
        SELECTED_RUN = SELECTED_RUN or choose_architecture()
        config = config_from(REPO_ROOT/candidate['config'],SELECTED_RUN)
        calibration = Path(STORAGE)/'calibration'/(calibration_key(config)+'.json')
        if not calibration.exists():
            ledger = Path(STORAGE)/'budget.json'
            spent = sum(j.get('elapsed_seconds',0) for j in read_json(ledger)['jobs']
                        if j['action']=='benchmark') if ledger.exists() else 0
            seconds = min(150,max(0,300-spent))
            if seconds<25:
                print(experiment_id,'omitido: sin margen de calibración')
                continue
            bench = run_action('benchmark',{'dataset':DATASET_ROOT,'config':config,
                'enable_optional':ENABLE_OPTIONAL},seconds)
            if bench['status']!='completado':
                print('Calibración no válida; no se entrena esta configuración.')
                continue
        trained = run_action('train',{'dataset':DATASET_ROOT,'config':config,
            'enable_optional':ENABLE_OPTIONAL,'resume':None},config['cap_seconds'])
        if trained['status']=='completado':
            validated = run_action('validate',{'dataset':DATASET_ROOT,'run':trained['run']},300)
            if validated['status']=='completado':
                SELECTED_RUN = choose_architecture() or SELECTED_RUN
        else:
            print('Entrenamiento no completo: no se promociona automáticamente.')
        budget_status()
else:
    print('Entrenamiento desactivado. Activa RUN_PIPELINE en la sección 2 para iniciar esta sección.')
'@
Markdown @'
## 7. Comparar y seleccionar resultados reales

Selecciona una COHORT entre las mostradas. Solo se ordenan resultados completos y
comparables; una diferencia pequeña con una semilla no demuestra significancia.
Curvas, métricas por clase, errores y slices están en las carpetas de validación
de Drive. No se consulta test para seleccionar.
'@
Code @'
COHORT = None
table = collect(STORAGE)
if table.empty:
    print('Todavía no hay entrenamientos guardados. No se generan métricas ficticias.')
else:
    display(table)
    print('Cohortes disponibles:',table.cohort.unique())
    if COHORT and 'score' in table:
        ranked = table[(table.cohort==COHORT)&table.comparable&table.score.notna()].sort_values(
            ['score','map50'],ascending=False)
        display(ranked)
        comparison_figures(ranked,Path(STORAGE)/'comparisons'/COHORT)
    SELECTED_RUN = SELECTED_RUN or choose_architecture()
    print('Candidato de validación interna:',SELECTED_RUN)
budget_status()
'@
Markdown '## 8. Inferencia opcional: checkpoint fijo y validación interna'
Code @'
RUN_INFERENCE = False
INFERENCE_OVERRIDES = {'tile_size':640,'tile_overlap':0.25,'tile_merge_iou':0.5}
if RUN_INFERENCE:
    if not SELECTED_RUN:
        raise ValueError('Primero seleccionar un entrenamiento completo y validado.')
    job = run_action('inference',{'dataset':DATASET_ROOT,'run':SELECTED_RUN,
        'inference_overrides':INFERENCE_OVERRIDES},300)
    if job['status']=='completado':
        SELECTED_INFERENCE_RUN = job['run']
        display(job['result']['metrics'])
else:
    print('Inferencia adicional desactivada. NMS se rechaza si no es operativo en salida end-to-end.')
'@
Markdown @'
## 9. Congelar selección, evaluar test una vez y generar ZIP

Activar las variables finales de la sección 2 solo tras seleccionar por validación
interna. Se congela checkpoint/configuración/umbral antes de test. El oculto solo
se predice, sin métricas supervisadas. El ZIP se verifica con el formateador oficial
y permanece en Drive. Se utiliza la reserva final del presupuesto.
'@
Code @'
frozen_path = Path(STORAGE)/'frozen_selection.json'
if FREEZE_FINAL_SELECTION:
    if not SELECTED_RUN:
        raise ValueError('Seleccionar primero un entrenamiento completo y validado.')
    protocol = load_protocol(PROTOCOL_PATH,DATASET_ROOT)
    if not frozen_path.exists():
        display(freeze_selection(SELECTED_RUN,protocol,frozen_path,inference_run=SELECTED_INFERENCE_RUN))
    else:
        print('Se conserva la selección ya congelada:',frozen_path)
if RUN_FINAL_TEST:
    display(run_action('test',{'dataset':DATASET_ROOT},240,final=True))
if RUN_SUBMISSION:
    submission = run_action('submission',{'dataset':DATASET_ROOT},240,final=True)
    display(submission)
    if submission['status']=='completado':
        print('ZIP en Drive:',submission['result']['zip'])
if not (FREEZE_FINAL_SELECTION or RUN_FINAL_TEST or RUN_SUBMISSION):
    print('Congelación, test y ZIP desactivados.')
'@
Markdown @'
## Referencias y verificaciones pendientes

Reto SpaceLAB y evaluador/formateador oficiales incluidos. Elgendy, *Deep Learning
for Vision Systems*, caps. 4, 6 y 7; Lakshmanan et al., *Practical Machine Learning
for Computer Vision*, caps. 3–8. Los hiperparámetros son hipótesis del proyecto.
[Drive API](https://developers.google.com/workspace/drive/api/guides/manage-downloads)
y [Colab](https://research.google.com/colaboratory/faq.html) documentan infraestructura.

Se ha preparado el archivo mediante escritura e inspección, sin ejecutar este
notebook, entrenar, consultar tu cuenta ni medir una T4. Permisos, versiones,
adaptadores y recuperación necesitan comprobación real. La disponibilidad de T4
depende de Google; instalar paquetes no convierte una sesión CPU en GPU.
El código y configuraciones están incrustados; no contiene imágenes, pesos ni
resultados. Las salidas guardadas después de ejecutar pueden incluir imágenes.
'@
$nb = @{nbformat=4;nbformat_minor=5;metadata=@{kernelspec=@{name='python3';language='python';display_name='Python 3'};
    language_info=@{name='python'};colab=@{name='SpaceLAB_Colab.ipynb';provenance=@()};accelerator='GPU'};cells=$cells.ToArray()}
$target = Join-Path $repoRoot 'SpaceLAB_Colab.ipynb'
[System.IO.File]::WriteAllText($target,($nb|ConvertTo-Json -Depth 40),$utf8)
Write-Output "Notebook único escrito: $target ($($cells.Count) celdas). Ninguna celda ejecutada."
