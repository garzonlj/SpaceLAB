"""Explicit future data staging. Never downloads a shared Drive folder on import."""
import json
import os
import shutil
import time
import uuid
import zipfile
import zlib
from pathlib import Path

from .io import dataset_root

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t"


class _Progress:
    """Real file/byte progress, plus a small status file outside the dataset."""
    def __init__(self, destination, phase, total_files=None, total_bytes=None):
        self.path = destination.parent/(destination.name+'.progress.json')
        self.phase = phase
        self.total_files, self.total_bytes = total_files, total_bytes
        self.files, self.bytes, self.reused = 0, 0, 0
        self.started, self.last_write = time.monotonic(), 0
        self.bars = []
        try:
            from tqdm.auto import tqdm
        except ImportError:
            pass
        else:
            self.bars.append(tqdm(total=total_files, desc=phase+' · archivos', unit='archivo', mininterval=1))
            if total_bytes is not None:
                self.bars.append(tqdm(total=total_bytes, desc=phase+' · datos', unit='B', unit_scale=True,
                                      unit_divisor=1024, mininterval=1))
        print(phase+' — progreso real de archivos y bytes; el reloj experimental no ha comenzado.',flush=True)
        self.write(force=True)

    def write(self, force=False, state='en curso', current=None):
        now = time.monotonic()
        if not force and now-self.last_write < 1:
            return
        elapsed = max(0,now-self.started)
        percent = 100*self.bytes/self.total_bytes if self.total_bytes else None
        snapshot = {'fase':self.phase,'estado':state,'archivos_completados':self.files,
            'archivos_total':self.total_files,'bytes_copiados':self.bytes,'bytes_total':self.total_bytes,
            'porcentaje_bytes':percent,'segundos_transcurridos':round(elapsed,1),'archivo_actual':current}
        snapshot['archivos_reutilizados'] = self.reused
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary = self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temporary,self.path)
        if not self.bars:
            total = str(self.total_files) if self.total_files is not None else '?'
            percentage = f' | {percent:.1f}%' if percent is not None else ''
            print(f'{self.phase}: {self.files}/{total} archivos{percentage} | {self.bytes/1024**2:.1f} MiB',flush=True)
        self.last_write = now

    def advance(self, byte_count=0, file_count=0, current=None):
        self.bytes += byte_count
        self.files += file_count
        if self.bars:
            self.bars[0].update(file_count)
            if len(self.bars)>1:
                self.bars[1].update(byte_count)
        self.write(current=current)

    def close(self, state):
        self.write(force=True,state=state)
        for bar in self.bars:
            bar.close()


def _copy_stream(reader, target, progress, name):
    target.parent.mkdir(parents=True,exist_ok=True)
    partial = target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')
    with partial.open('wb') as writer:
        for chunk in iter(lambda: reader.read(1024*1024),b''):
            writer.write(chunk)
            progress.advance(byte_count=len(chunk),current=name)
    os.replace(partial,target)
    progress.advance(file_count=1,current=name)


def verify_layout(path):
    root = dataset_root(path)
    for split,expected in [("train",5926),("test",801),("oculto",1743)]:
        count = len(list((root/"images"/split).glob("*.jpg")))
        if count != expected:
            raise ValueError(f"Copia incompleta: {split} tiene {count}, se esperaban {expected}. No se sobrescribe automáticamente.")
        if split != 'oculto':
            image_ids = {p.stem for p in (root/'images'/split).glob('*.jpg')}
            label_ids = {p.stem for p in (root/'labels'/split).glob('*.txt')}
            if image_ids != label_ids:
                raise ValueError(f'Copia incompleta: faltan o sobran etiquetas de {split}.')
    return root


def stage_dataset(source, destination, source_zip=None, resume=False):
    """Copy from a mounted Drive shortcut, or extract a user-selected local ZIP.

    Run BEFORE opening the timed session. This is data preparation, not a
    benchmark. Originals stay intact. resume=True repairs a partial local copy
    without deleting it. Existing files are reused only with matching metadata
    (directory copy) or a verified CRC32 (ZIP extraction).
    """
    source,destination = Path(source).expanduser(),Path(destination).expanduser()
    if destination.exists() and not resume:
        return verify_layout(destination)
    if source_zip:
        # Extract only into a fresh directory and reject path traversal.
        with zipfile.ZipFile(source_zip) as archive:
            for member in archive.infolist():
                target = (destination/member.filename).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise ValueError(f"Ruta fuera del destino en ZIP: {member.filename}")
            members = [member for member in archive.infolist() if not member.is_dir()]
            destination.mkdir(parents=True,exist_ok=resume)
            progress = _Progress(destination,'Extrayendo ZIP',len(members),sum(m.file_size for m in members))
            try:
                for member in members:
                    target = destination/member.filename
                    if resume and target.is_file() and target.stat().st_size == member.file_size:
                        crc = 0
                        with target.open('rb') as existing:
                            for chunk in iter(lambda: existing.read(1024*1024),b''):
                                crc = zlib.crc32(chunk,crc)
                        if crc & 0xffffffff == member.CRC:
                            progress.reused += 1
                            progress.advance(byte_count=member.file_size,file_count=1,current=member.filename)
                            continue
                    with archive.open(member) as reader:
                        _copy_stream(reader,target,progress,member.filename)
            except BaseException:
                progress.close('interrumpido')
                raise
            else:
                progress.close('copia terminada; verificando estructura')
    else:
        root = dataset_root(source)
        if root.resolve() == destination.resolve() or root.resolve() in destination.resolve().parents:
            raise ValueError('El destino de copia debe estar fuera de la carpeta original.')
        inventory = _Progress(destination,'Contando archivos de Drive')
        try:
            files = []
            for path in root.rglob('*'):
                if path.is_file():
                    stat = path.stat()
                    files.append((path,stat.st_size,stat.st_mtime_ns))
                    inventory.advance(file_count=1,current=path.name)
        except BaseException:
            inventory.close('interrumpido')
            raise
        inventory.close('inventario terminado')
        destination.mkdir(parents=True,exist_ok=resume)
        progress = _Progress(destination,'Copiando Drive a Colab',len(files),sum(size for _,size,_ in files))
        try:
            for path,size,mtime_ns in files:
                relative = path.relative_to(root)
                target = destination/relative
                if resume and target.is_file():
                    stat = target.stat()
                    if stat.st_size == size and stat.st_mtime_ns == mtime_ns:
                        progress.reused += 1
                        progress.advance(byte_count=size,file_count=1,current=relative.as_posix())
                        continue
                with path.open('rb') as reader:
                    _copy_stream(reader,target,progress,relative.as_posix())
                shutil.copystat(path,target)
        except BaseException:
            progress.close('interrumpido')
            raise
        else:
            progress.close('copia terminada; verificando estructura')
    try:
        result = verify_layout(destination)
    except BaseException:
        progress.close('verificación fallida')
        raise
    progress.close('completado y verificado')
    print('Dataset listo:',result,flush=True)
    return result
