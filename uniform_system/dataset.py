import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
import cv2
import yaml
from .core import CLASS_NAMES, validate_class_map

REQUIRED_COLUMNS={"image","label","person_ids","session_id","source_video"}

def read_manifest(path):
    path=Path(path)
    with path.open(newline="") as f:
        reader=csv.DictReader(f)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(REQUIRED_COLUMNS)}")
        rows=list(reader)
    if not rows: raise ValueError("Manifest has no images")
    seen=set()
    for row in rows:
        if not row["image"] or not row["label"] or not row["session_id"]:
            raise ValueError("Every row needs image, label and session_id")
        resolved=str((path.parent/row["image"]).resolve())
        if resolved in seen: raise ValueError("Duplicate image in manifest")
        seen.add(resolved)
    return rows

def group_split(rows,seed=42):
    """Connected components link shared people, sessions OR videos transitively."""
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    keys={}
    for i,row in enumerate(rows):
        groups=[("person",p.strip()) for p in row["person_ids"].split(";") if p.strip()]
        groups += [(k,row[k]) for k in ("session_id","source_video") if row[k]]
        for key in groups:
            if key in keys: parent[find(i)]=find(keys[key])
            else: keys[key]=i
    components={}
    for i in range(len(rows)): components.setdefault(find(i),[]).append(i)
    groups=list(components.values()); random.Random(seed).shuffle(groups)
    groups.sort(key=len,reverse=True)
    target=dict(train=.70*len(rows),val=.15*len(rows),test=.15*len(rows))
    result={k:[] for k in target}
    for group in groups:
        # Minimize total squared deviation after each whole-group assignment.
        dest=min(target,key=lambda s:sum((len(result[k])+(len(group) if k==s else 0)-target[k])**2 for k in target))
        result[dest].extend(group)
    return result

def label_counts(label_path):
    counts=Counter()
    for number,line in enumerate(Path(label_path).read_text().splitlines(),1):
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=5: raise ValueError(f"{label_path}:{number}: expected class x y w h")
        try:
            cid=int(fields[0]); x,y,w,h=map(float,fields[1:])
        except ValueError as exc: raise ValueError(f"{label_path}:{number}: invalid numeric label") from exc
        if cid not in range(7) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
            raise ValueError(f"{label_path}:{number}: invalid class or normalized geometry")
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
            raise ValueError(f"{label_path}:{number}: bounding box extends outside image")
        counts[cid]+=1
    return counts

def validate_manifest(path):
    rows=read_manifest(path); base=Path(path).parent; counts=Counter(); errors=[]
    for row in rows:
        image=base/row["image"]; label=base/row["label"]
        try:
            if not image.is_file() or cv2.imread(str(image)) is None: raise ValueError(f"Invalid image: {image}")
            counts.update(label_counts(label))
        except (ValueError,OSError) as exc: errors.append(str(exc))
    report={"images":len(rows),"class_counts":{n:counts[i] for i,n in enumerate(CLASS_NAMES)},"errors":errors}
    if errors: raise ValueError(json.dumps(report,indent=2))
    return report

def split_dataset(manifest,output,seed=42):
    report=validate_manifest(manifest); rows=read_manifest(manifest); base=Path(manifest).parent
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError("Split output must be empty to avoid stale data/leakage")
    output.mkdir(parents=True,exist_ok=True); assignments=group_split(rows,seed); final=[]
    # Reject identical images across nominally unrelated groups too.
    hashes={}
    for split,indices in assignments.items():
        for idx in indices:
            digest=hashlib.sha256((base/rows[idx]["image"]).read_bytes()).hexdigest()
            if digest in hashes and hashes[digest]!=split:
                raise ValueError("Identical image content assigned to different splits; repair manifest grouping")
            hashes[digest]=split
    for split,indices in assignments.items():
        for folder in ("images","labels"): (output/folder/split).mkdir(parents=True,exist_ok=True)
        for idx in indices:
            row=rows[idx]; stem=f"{idx:06d}"
            image=f"images/{split}/{stem}{Path(row['image']).suffix.lower()}"; label=f"labels/{split}/{stem}.txt"
            shutil.copy2(base/row["image"],output/image); shutil.copy2(base/row["label"],output/label)
            final.append(dict(row,image=image,label=label,split=split))
    with (output/"manifest.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[*(key for key in rows[0] if key!="split"),"split"]); writer.writeheader(); writer.writerows(final)
    data=dict(path=str(output.resolve()),train="images/train",val="images/val",test="images/test",names=dict(enumerate(CLASS_NAMES)))
    (output/"dataset.yaml").write_text(yaml.safe_dump(data,sort_keys=False))
    report["split_class_counts"]={split:{name:sum(label_counts(output/"labels"/split/(f"{idx:06d}.txt"))[cid] for idx in indices)
        for cid,name in enumerate(CLASS_NAMES)} for split,indices in assignments.items()}
    report.update(seed=seed,split_counts={k:len(v) for k,v in assignments.items()},
                  note="Whole connected groups preserved; exact 70/15/15 may be impossible. Empty splits must be repaired before training.")
    (output/"distribution.json").write_text(json.dumps(report,indent=2))
    return report

def extract_frames(video,output,interval=1.0,session="",people=""):
    if interval<=0 or not session: raise ValueError("Positive interval and consented session ID required")
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError("Extraction output must be empty")
    out.mkdir(parents=True,exist_ok=True); (out/"images").mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(video)); rows=[]
    try:
        if not cap.isOpened(): raise ValueError("Cannot open source video")
        fps=cap.get(cv2.CAP_PROP_FPS)
        if fps<=0: raise ValueError("Source video has no valid frame rate")
        n=0; next_time=0.0
        while True:
            ok,frame=cap.read()
            if not ok:
                if n==0: raise ValueError("Video has no readable frames")
                if cap.get(cv2.CAP_PROP_FRAME_COUNT)>n+1: raise IOError("Premature video read failure")
                break
            if n/fps>=next_time:
                image=f"images/frame_{n:08d}.jpg"
                if not cv2.imwrite(str(out/image),frame): raise IOError("Frame write failed")
                rows.append(dict(image=image,label=f"labels/frame_{n:08d}.txt",person_ids=people,
                                 session_id=session,source_video=str(Path(video).resolve())))
                next_time+=interval
            n+=1
    finally: cap.release()
    with (out/"manifest.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["image","label","person_ids","session_id","source_video"])
        writer.writeheader(); writer.writerows(rows)
    return {"extracted_frames":len(rows),"manifest":str(out/"manifest.csv"),"note":"Labels are NOT generated. Crop people and annotate before validation."}

def resolve_dataset(path):
    path=Path(path).resolve(); data=yaml.safe_load(path.read_text()); validate_class_map(data["names"])
    base=Path(data.get("path",path.parent))
    if not base.is_absolute(): base=Path.cwd()/base
    data["path"]=str(base.resolve())
    for split in ("train","val","test"):
        entry=data.get(split)
        if not isinstance(entry,str): raise ValueError(f"Dataset needs a directory for {split}")
        folder=base/entry
        if not folder.is_dir() or not any(p.suffix.lower() in (".jpg",".jpeg",".png",".bmp") for p in folder.iterdir()):
            raise ValueError(f"Missing/empty {split} images: {folder}")
        for image in folder.iterdir():
            if image.suffix.lower() not in (".jpg",".jpeg",".png",".bmp"): continue
            label=base/"labels"/split/(image.stem+".txt")
            label_counts(label)  # Missing labels fail; reviewed empty labels are permitted.
            if cv2.imread(str(image)) is None: raise ValueError(f"Invalid image: {image}")
    return data

def prepare_clahe(data,output):
    from .pipeline import preprocess,read_image,save_image
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError("CLAHE output must be empty")
    base=Path(data["path"])
    for split in ("train","val","test"):
        images=base/data[split]
        for image in images.iterdir():
            if image.suffix.lower() not in (".jpg",".jpeg",".png",".bmp"): continue
            relative=image.relative_to(base)
            save_image(output/relative,preprocess(read_image(image),"clahe"))
            label=base/"labels"/split/(image.stem+".txt")
            dest=output/"labels"/split/label.name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(label,dest)
    manifest=base/"manifest.csv"
    if manifest.exists(): shutil.copy2(manifest,output/"manifest.csv")
    return dict(data,path=str(output.resolve()))
