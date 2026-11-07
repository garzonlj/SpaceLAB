"""Explicit Colab data preparation from the team's shared Google Drive folder.

No authentication, listing or downloading occurs at import time. Never writes to
the source Drive folder. Downloads happen before the experimental time budget.
"""
import hashlib
import os
from collections import deque
from pathlib import Path

from .io import dataset_root
from .preparation import stage_dataset, verify_layout

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"


def prepare_from_drive(folder_id, destination, mounted_source=None, zip_file_id=None):
    """Use a mounted copy when available, otherwise authenticated Drive API.

    Accept a dataset ZIP or an uncompressed dataset folder (possibly nested).
    ZIP is preferred over thousands of individual API requests. A directory copy
    can resume completed files; ZIP extraction uses the existing safe staging code.
    """
    destination = Path(destination)
    if destination.exists():
        return verify_layout(destination)
    if mounted_source and Path(mounted_source).exists():
        source = Path(mounted_source)
        try:
            root = dataset_root(source)
        except ValueError:
            archives = list(source.glob('*spacelab*dataset*.zip'))
            if len(archives) == 1:
                return stage_dataset(source, destination, source_zip=archives[0])
        else:
            return stage_dataset(root, destination)

    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    print('Autorizando lectura de la carpeta compartida de Drive...', flush=True)
    auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)

    def children(parent):
        result, token = [], None
        while True:
            response = service.files().list(
                q=f"'{parent}' in parents and trashed=false", pageSize=1000,
                pageToken=token, supportsAllDrives=True, includeItemsFromAllDrives=True,
                fields='nextPageToken,files(id,name,mimeType,size,md5Checksum,shortcutDetails)'
            ).execute(num_retries=3)
            result.extend(response.get('files', []))
            token = response.get('nextPageToken')
            if not token:
                return result

    def resolve(item):
        if item['mimeType'] == SHORTCUT_MIME:
            return service.files().get(fileId=item['shortcutDetails']['targetId'], supportsAllDrives=True,
                fields='id,name,mimeType,size,md5Checksum').execute(num_retries=3)
        return item

    completed = [0]

    def download(item, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and 'size' in item and path.stat().st_size == int(item['size']):
            if not item.get('md5Checksum'):
                return
            h = hashlib.md5()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    h.update(chunk)
            if h.hexdigest() == item['md5Checksum']:
                return
        partial = path.with_name(path.name+'.part')
        request = service.files().get_media(fileId=item['id'], supportsAllDrives=True)
        with partial.open('wb') as stream:
            transfer = MediaIoBaseDownload(stream, request, chunksize=16*1024*1024)
            done, previous = False, -1
            while not done:
                status, done = transfer.next_chunk(num_retries=3)
                if status and int(item.get('size', 0)) > 100*1024*1024:
                    percent = int(100*status.progress())
                    if percent//10 != previous:
                        print(f'{item["name"]}: {percent}%', flush=True)
                        previous = percent//10
        if 'size' in item and partial.stat().st_size != int(item['size']):
            raise IOError('Descarga incompleta: '+item['name'])
        if item.get('md5Checksum'):
            h = hashlib.md5()
            with partial.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    h.update(chunk)
            if h.hexdigest() != item['md5Checksum']:
                raise IOError('La suma MD5 de Drive no coincide: '+item['name'])
        os.replace(partial, path)
        completed[0] += 1
        if completed[0] % 200 == 0:
            print(f'{completed[0]} archivos descargados', flush=True)

    metadata = service.files().get(fileId=zip_file_id or folder_id, supportsAllDrives=True,
        fields='id,name,mimeType,size,md5Checksum').execute(num_retries=3)
    if metadata['mimeType'] != FOLDER_MIME:
        archives, roots = [metadata], []
    else:
        queue, seen, archives, roots = deque([(metadata,0)]), set(), [], []
        while queue:
            folder, depth = queue.popleft()
            if folder['id'] in seen:
                continue
            seen.add(folder['id'])
            entries = [resolve(item) for item in children(folder['id'])]
            names = {e['name'] for e in entries}
            if {'data.yaml','images','labels'} <= names:
                roots.append(folder)
                continue  # Do not enumerate images during source discovery.
            archives.extend(e for e in entries if e['name'].lower().endswith('.zip')
                            and 'dataset' in e['name'].lower())
            if depth < 3:
                queue.extend((e,depth+1) for e in entries if e['mimeType']==FOLDER_MIME
                             and ('spacelab' in e['name'].lower() or 'dataset' in e['name'].lower()))
    if len(archives) == 1:
        archive = archives[0]
        if not archive['name'].lower().endswith('.zip'):
            raise ValueError('Seleccione un archivo ZIP del dataset, no otro documento.')
        local_zip = destination.parent/'spacelab_dataset_download.zip'
        download(archive,local_zip)
        return stage_dataset(destination.parent,destination,source_zip=local_zip)
    if len(archives) > 1:
        choices = [(a['name'],a['id']) for a in archives]
        raise ValueError(f'Hay varios ZIP. Configure ZIP_FILE_ID con el correcto: {choices}')
    if len(roots) != 1:
        raise ValueError('No se encontró un único dataset o ZIP. Revise acceso a la carpeta y DATASET_FOLDER_ID.')

    # Keep partially downloaded files outside the final destination so a retry
    # never mistakes a partial directory for a completed dataset.
    staging = destination.with_name(destination.name+'_drive_download')
    staging.mkdir(parents=True,exist_ok=True)
    print('Dataset sin ZIP: descargando por archivos; puede tardar más que un ZIP.', flush=True)

    def copy_folder(remote_id, local, active):
        if remote_id in active:
            raise ValueError('Ciclo de accesos directos en la carpeta de Drive')
        active = active | {remote_id}
        local.mkdir(parents=True,exist_ok=True)
        for entry in children(remote_id):
            name = entry['name']
            if name in ('.','..') or '/' in name or '\\' in name:
                raise ValueError('Nombre de archivo no portable en Drive: '+name)
            target = local/name
            item = resolve(entry)
            if item['mimeType'] == FOLDER_MIME:
                copy_folder(item['id'],target,active)
            elif item['mimeType'].startswith('application/vnd.google-apps.'):
                raise ValueError('Documento de Google inesperado dentro del dataset: '+name)
            else:
                download(item,target)
    root_entries = children(roots[0]['id'])
    for entry in root_entries:
        if entry['name'] in ('images','labels') and entry['mimeType']==FOLDER_MIME:
            copy_folder(entry['id'],staging/entry['name'],set())
        elif entry['name']=='data.yaml':
            download(entry,staging/'data.yaml')
    verify_layout(staging)
    staging.rename(destination)
    return verify_layout(destination)
