from dataclasses import asdict
from pathlib import Path
import time
import json
import textwrap
import cv2
import numpy as np
from .core import Detection, assess, validate_class_map, crop_to_original, validate_evidence

SHORT_REASONS = {
    "ALL_REQUIREMENTS_CONFIRMED":"All enabled clothing components confirmed",
    "NON_UNIFORM_UPPER":"Confirmed upper garment violation",
    "NON_UNIFORM_LOWER":"Confirmed lower garment violation",
    "REQUIRED_COMPONENT_UNCONFIRMED":"Required clothing unconfirmed",
    "LOW_CONFIDENCE":"Low-confidence clothing evidence",
    "CONFLICTING_DETECTIONS":"Conflicting clothing predictions",
    "AMBIGUOUS_PERSON_ASSOCIATION":"Clothing ownership ambiguous",
    "SMALL_REGION":"Region too small to assess",
    "INVALID_GARMENT_LOCATION":"Clothing in implausible body region",
    "CROPPED_GARMENT":"Garment partially cropped",
}
COLORS = {"Compliant":(30,180,30), "Non-compliant":(30,30,230), "Uncertain":(0,180,255)}

class UniformPipeline:
    def __init__(self, clothing_weights, rules, person_weights="yolo11n.pt", device="auto", metadata=None):
        path=Path(clothing_weights)
        if not path.is_file():
            raise FileNotFoundError("Custom clothing weights missing. Train with `uniform train` or supply a seven-class checkpoint; generic weights are not a uniform model.")
        sidecar=Path(metadata) if metadata else path.with_suffix(".metadata.json")
        if not sidecar.is_file():
            raise ValueError("Model preprocessing metadata missing. Supply WEIGHTS.metadata.json (see docs/model_metadata.example.json).")
        info=json.loads(sidecar.read_text())
        if info.get("preprocessing") != rules.preprocessing:
            raise ValueError("Training/inference preprocessing mismatch")
        from ultralytics import YOLO
        import torch
        self.device = (0 if torch.cuda.is_available() else "cpu") if device=="auto" else device
        self.clothing=YOLO(str(path),task="detect")
        validate_class_map(self.clothing.names)
        if self.clothing.task != "detect":
            raise ValueError("A detection model is required")
        self.person=YOLO(str(person_weights),task="detect")
        ids=[int(k) for k,v in self.person.names.items() if v=="person"]
        if len(ids)!=1:
            raise ValueError("Person detector must contain exactly one 'person' class")
        self.person_id=ids[0]; self.rules=rules

    def process(self, frame, frame_number=0, timestamp_seconds=0):
        if frame is None or frame.ndim!=3 or frame.shape[2]!=3 or frame.size==0:
            raise ValueError("Input must be a nonempty BGR image")
        start=time.perf_counter(); h,w=frame.shape[:2]
        preds=self.person.predict(frame,classes=[self.person_id],conf=self.rules.person_confidence,
                                  imgsz=self.rules.imgsz,device=self.device,verbose=False)[0]
        persons=[]; person_conf=[]
        for b,c in zip(preds.boxes.xyxy.cpu().tolist(),preds.boxes.conf.cpu().tolist()):
            x1,y1,x2,y2=b
            # Integer crop boundaries are also the coordinate mapping origin.
            box=(max(0,int(x1)),max(0,int(y1)),min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2))))
            if box[2]>box[0] and box[3]>box[1]:
                persons.append(box); person_conf.append(c)
        output=[]
        for index,p in enumerate(persons):
            crop=frame[p[1]:p[3],p[0]:p[2]]
            crop=preprocess(crop,self.rules.preprocessing)
            result=self.clothing.predict(crop,conf=self.rules.candidate_confidence,imgsz=self.rules.imgsz,
                                         device=self.device,verbose=False)[0]
            detections=[]
            for b,c,k in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                d=Detection(int(k),c,crop_to_original(b,p))
                detections.append(validate_evidence(d,index,persons,self.rules))
            # Overlapping alternative labels remain visible; NMS is deliberately class-aware.
            judgment=assess(detections,self.rules)
            output.append(dict(person_index=index, temporary_track_id=None, person_box_xyxy=list(p),
                               person_confidence=person_conf[index],detections=[d.export() for d in detections],
                               **asdict(judgment)))
        latency=time.perf_counter()-start
        record=dict(frame_number=frame_number,timestamp_seconds=timestamp_seconds,
                    coordinate_space="original_image_pixels_xyxy",width=w,height=h,persons=output,
                    message="No person detected." if not output else "",latency_seconds=latency,
                    processing_fps=1/latency if latency else 0)
        return draw(frame,record),record

def preprocess(image, method):
    # OpenCV BGR input is accepted directly by Ultralytics. It performs RGB conversion,
    # aspect-preserving letterbox, CHW conversion and /255 normalization internally.
    if method=="baseline": return image
    if method!="clahe": raise ValueError("Unknown preprocessing")
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

def draw(frame,record):
    out=frame.copy()
    if not record["persons"]:
        cv2.putText(out,"No person detected.",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,180,255),2)
    for p in record["persons"]:
        color=COLORS[p["status"]]; x1,y1,x2,y2=map(int,p["person_box_xyxy"])
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f'P{p["person_index"]}: {p["status"]}',(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        # Short readable explanation on the image; full reasons are in UI/exports.
        short=SHORT_REASONS[p["reason_codes"][0]]
        for line_index,line in enumerate(textwrap.wrap(short,width=max(12,(out.shape[1]-x1)//7))):
            y=min(out.shape[0]-5,max(20,y1+20)+line_index*15)
            cv2.putText(out,line,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        for d in p["detections"]:
            a,b,c,e=map(int,d["box_xyxy"]); dc=(0,180,255) if d["issues"] else color
            cv2.rectangle(out,(a,b),(c,e),dc,1)
            cv2.putText(out,f'{d["class_name"]} {d["confidence"]:.2f}',(a,max(12,b)),cv2.FONT_HERSHEY_SIMPLEX,.35,dc,1)
    return out

def read_image(path):
    try:
        raw=np.frombuffer(Path(path).read_bytes(),dtype=np.uint8)
    except OSError as exc: raise ValueError(f"Cannot read image: {path}") from exc
    frame=cv2.imdecode(raw,cv2.IMREAD_COLOR) if raw.size else None
    if frame is None: raise ValueError(f"Invalid or empty image: {path}")
    return frame

def save_image(path,frame):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),frame): raise IOError(f"Cannot save image: {path}")

def video_frames(pipeline, source, output=None, max_frames=0, stop=None):
    """Generator releases capture/writer in finally; callers must close it on early stop."""
    cap=cv2.VideoCapture(source); writer=None
    try:
        if not cap.isOpened(): raise ValueError(f"Cannot open video/webcam: {source}")
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0: fps=25.0
        count=0; started=time.perf_counter()
        while not max_frames or count<max_frames:
            if stop and stop(): break
            ok,frame=cap.read()
            if not ok:
                if count==0: raise ValueError("Video/webcam returned no readable frames")
                # Early EOF on a file with a declared frame count is a read failure.
                expected=cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not isinstance(source,int) and expected>count+1:
                    raise IOError(f"Video read failed at frame {count} of {int(expected)}")
                break
            timestamp=(time.perf_counter()-started) if isinstance(source,int) else count/fps
            annotated,record=pipeline.process(frame,count,timestamp)
            record["timestamp_source"]="monotonic_elapsed" if isinstance(source,int) else "frame_number/source_fps"
            record["source_fps"]=fps
            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True,exist_ok=True)
                    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(frame.shape[1],frame.shape[0]))
                    if not writer.isOpened(): raise IOError(f"Cannot open video writer: {output}")
                writer.write(annotated)
            count+=1
            yield annotated,record
    finally:
        cap.release()
        if writer: writer.release()
