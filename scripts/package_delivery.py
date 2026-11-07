"""Copy the completed, verified competition submission into a simple delivery folder."""
import argparse
import shutil
import sys
from pathlib import Path

for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,"reconfigure"):
        stream.reconfigure(encoding="utf-8",errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from spacelab.io import dataset_root, digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--output", default=str(ROOT/"artifacts/entrega_spacelab"))
    args = parser.parse_args()
    storage,output = Path(args.storage).resolve(),Path(args.output).resolve()
    completion = read_json(storage/"pipeline_completed.json")
    state = read_json(storage/"pipeline_state.json")
    frozen = read_json(storage/"frozen_selection.json")
    submission = state["jobs"]["submission"]
    test = state["jobs"]["official_test"]
    if submission["status"] != "completado" or test["status"] != "completado":
        raise RuntimeError("La entrega y evaluación final aún no están completas.")
    source_zip = Path(submission["result"]["zip"])
    output.mkdir(parents=True,exist_ok=True)
    target = output/"SpaceLAB_entrega.zip"
    shutil.copy2(source_zip,target)
    # The official verifier checks filenames, coordinates, classes and confidence.
    from spacelab.evaluation import official_modules
    _,formatter = official_modules()
    root = dataset_root(state["dataset"])
    if not formatter.verificar(str(target),str(root/"images/oculto")):
        raise RuntimeError("El ZIP copiado no superó el verificador oficial.")
    validation = read_json(Path(completion["best_inference"])/"metrics/official_score.json")
    test_metrics = read_json(Path(test["run"])/"metrics/official_score.json")
    write_json(output/"metricas_validacion.json",validation)
    write_json(output/"metricas_test.json",test_metrics)
    write_json(output/"seleccion_congelada.json",frozen)
    shutil.copy2(frozen["checkpoint"],output/"modelo_best.pt")
    manifest = {"submission_zip":str(target),"sha256":digest(target),
                "bytes":target.stat().st_size,"official_verifier_passed":True,
                "validation_score":validation["total"]["score"],"test_score":test_metrics["total"]["score"],
                "training_run":completion["selected_run"],"inference_run":completion["best_inference"]}
    runtime = read_json(Path(test["run"])/"metrics/runtime.json")
    manifest["final_inference_environment"] = runtime["environment"]
    search_path = storage/"inference_search.json"
    if search_path.exists():
        shutil.copy2(search_path,output/"pruebas_inferencia.json")
    write_json(output/"delivery_manifest.json",manifest)
    lines = ["# Entrega SpaceLAB", "", "Subir únicamente **SpaceLAB_entrega.zip** al hackatón.", "",
             "El ZIP contiene predicciones para las 1.743 imágenes del conjunto oculto y umbral.txt.",
             "Pasó el verificador oficial del kit.", "", "| Conjunto | Score | mAP50–95 | F1 macro |",
             "|---|---:|---:|---:|"]
    for name,metrics in (("Validación interna",validation),("Test oficial local",test_metrics)):
        row = metrics["total"]
        lines.append(f"| {name} | {row['score']:.6f} | {row['map50_95']:.6f} | {row['f1_macro']:.6f} |")
    lines.extend(["", "El puntaje del ranking oculto todavía no se conoce; los valores anteriores son locales.",
                  "Modelo, umbral y receta quedaron congelados antes de consultar test.", "",
                  "El modelo se entrenó en la GTX 1650. La inferencia final se realizó en CPU por un bloqueo del entorno CUDA.",
                  "Se entrega la mejor receta confirmada en validación completa. Las pruebas de inferencia interrumpidas no se consideran mejoras.",
                  "Consultar pruebas_inferencia.json para el estado de cada intento.", "",
                  "modelo_best.pt es un checkpoint del pipeline SpaceLAB, no un archivo para cargar directamente con YOLO().",
                  "Usar src/spacelab/checkpointing.py y la configuración guardada para reproducirlo.", "",
                  "SHA-256 del ZIP: `"+manifest["sha256"]+"`", ""])
    (output/"LEEME.md").write_text("\n".join(lines),encoding="utf-8")
    print("Entrega verificada:",target,flush=True)
    print("SHA-256:",manifest["sha256"],flush=True)


if __name__ == "__main__":
    main()
