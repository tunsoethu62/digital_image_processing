import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import hashlib
import yaml
from .core import validate_class_map
from .dataset import resolve_dataset

def device_name(device):
    import torch
    return (0 if torch.cuda.is_available() else "cpu") if device=="auto" else device

def environment():
    result={"python":platform.python_version(),"platform":platform.platform()}
    for name in ("ultralytics","torch","opencv-python","numpy","PyYAML","streamlit"):
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]="not installed"
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(),hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "CPU")
    except ImportError: pass
    return result

def train(args):
    from ultralytics import YOLO
    data=resolve_dataset(args.data)
    if args.preprocessing=="clahe":
        from .dataset import prepare_clahe
        data=prepare_clahe(data,Path(args.project)/f"{args.name}_clahe_data")
    record_dir=Path(args.project)/f"{args.name}_config"
    if record_dir.exists(): raise ValueError("Run configuration already exists; choose a new run name")
    record_dir.mkdir(parents=True)
    resolved=record_dir/"dataset.yaml"; resolved.write_text(yaml.safe_dump(data,sort_keys=False))
    for key in ("epochs","batch","imgsz"):
        if getattr(args,key)<=0: raise ValueError(f"{key} must be positive")
    model=YOLO(args.checkpoint,task="detect")
    model.train(data=str(resolved.resolve()),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
                seed=args.seed,device=device_name(args.device),project=args.project,name=args.name,
                exist_ok=False,plots=True)
    run=Path(model.trainer.save_dir)
    metadata=dict(preprocessing=args.preprocessing,classes=list(model.names.values()),
                  checkpoint=args.checkpoint,dataset=data,environment=environment(),
                  settings={k:getattr(args,k) for k in ("epochs","batch","imgsz","seed","device")})
    validate_class_map(model.names)
    for checkpoint in (run/"weights").glob("*.pt"):
        checkpoint.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2))
    (run/"project_record.json").write_text(json.dumps(metadata,indent=2))
    manifest=Path(data["path"])/"manifest.csv"
    if manifest.exists(): shutil.copy2(manifest,run/"dataset_manifest.csv")
    # Fingerprint exactly the split image/label bytes used in this training run.
    fingerprints={}
    base=Path(data["path"])
    for folder in ("images","labels"):
        for file in sorted((base/folder).rglob("*")):
            if file.is_file(): fingerprints[str(file.relative_to(base))]=hashlib.sha256(file.read_bytes()).hexdigest()
    (run/"dataset_sha256.json").write_text(json.dumps(fingerprints,indent=2))
    return {"run":str(run),"note":"Ultralytics writes checkpoints, results.csv and available curves in this run directory."}

def evaluate_detection(args):
    from ultralytics import YOLO
    data=resolve_dataset(args.data)
    weights=Path(args.weights)
    if not weights.is_file(): raise FileNotFoundError("Custom weights missing")
    sidecar=weights.with_suffix(".metadata.json")
    metadata=json.loads(sidecar.read_text())
    if metadata.get("preprocessing")!=args.preprocessing: raise ValueError("Model/evaluation preprocessing mismatch")
    if args.preprocessing=="clahe":
        from .dataset import prepare_clahe
        data=prepare_clahe(data,Path(args.project)/f"{args.name}_clahe_data")
    destination=Path(args.project)/f"{args.name}_config"; destination.mkdir(parents=True,exist_ok=True)
    resolved=destination/"dataset.yaml"; resolved.write_text(yaml.safe_dump(data,sort_keys=False))
    model=YOLO(str(weights)); validate_class_map(model.names)
    artifacts={}
    model.add_callback("on_val_end",lambda validator: artifacts.update(path=str(validator.save_dir)))
    results=model.val(data=str(resolved.resolve()),split=args.split,imgsz=args.imgsz,device=device_name(args.device),
                      project=args.project,name=args.name,plots=True)
    ids=results.box.ap_class_index.tolist()
    report={"split":args.split,"map50":float(results.box.map50),"map50_95":float(results.box.map),
            "per_class":{model.names[int(cid)]:dict(precision=float(results.box.p[i]),recall=float(results.box.r[i]),
                          map50=float(results.box.ap50[i]),map50_95=float(results.box.ap[i])) for i,cid in enumerate(ids)},
            "unrepresented_classes":[n for k,n in model.names.items() if k not in ids],
            "environment":environment(),"artifacts":artifacts.get("path","Unavailable: validator did not report its output directory")}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)); return report
