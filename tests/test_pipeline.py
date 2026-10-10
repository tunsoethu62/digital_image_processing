import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from uniform_system.core import Rules
from uniform_system.pipeline import UniformPipeline,read_image,video_frames,preprocess
from uniform_system.exports import PredictionWriter

class Array:
    def __init__(self,values): self.values=values
    def cpu(self): return self
    def tolist(self): return self.values
class FakeDetector:
    def __init__(self,boxes,classes,confidence): self.boxes=boxes; self.classes=classes; self.confidence=confidence
    def predict(self,*args,**kwargs):
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Array(self.boxes),cls=Array(self.classes),conf=Array(self.confidence)))]
def pipeline(people):
    p=UniformPipeline.__new__(UniformPipeline); p.rules=Rules(); p.device="cpu";p.person_id=0
    p.person=FakeDetector(people,[0]*len(people),[.99]*len(people))
    p.clothing=FakeDetector([[10,30,70,100],[10,130,70,220]],[0,2],[.99,.99])
    return p

def test_empty_scene_and_mapped_detections():
    frame=np.zeros((400,400,3),dtype=np.uint8)
    _,record=pipeline([]).process(frame)
    assert record["message"]=="No person detected." and record["persons"]==[]
    _,record=pipeline([[100,50,200,300]]).process(frame)
    person=record["persons"][0]
    assert person["detections"][0]["box_xyxy"]==[110,80,170,150]
    assert person["status"]=="Compliant" and person["temporary_track_id"] is None

def test_overlap_ownership_is_uncertain():
    frame=np.zeros((400,400,3),dtype=np.uint8)
    _,record=pipeline([[100,50,200,300],[120,50,220,300]]).process(frame)
    assert all(p["status"]=="Uncertain" for p in record["persons"])
    assert all("AMBIGUOUS_PERSON_ASSOCIATION" in p["reason_codes"] for p in record["persons"])

def test_missing_weights_and_invalid_input(tmp_path):
    with pytest.raises(FileNotFoundError,match="Custom clothing weights missing"):
        UniformPipeline(tmp_path/"missing.pt",Rules())
    p=tmp_path/"invalid.jpg"; p.write_text("not an image")
    with pytest.raises(ValueError): read_image(p)
    with pytest.raises(ValueError): pipeline([]).process(None)
    arr=np.zeros((20,20,3),dtype=np.uint8)
    assert preprocess(arr,"baseline") is arr
    assert preprocess(arr,"clahe").shape==arr.shape

def test_video_export_and_release(tmp_path):
    input_path=tmp_path/"input.avi"
    writer=cv2.VideoWriter(str(input_path),cv2.VideoWriter_fourcc(*"MJPG"),10,(100,100))
    if not writer.isOpened(): pytest.skip("MJPG codec unavailable")
    for i in range(3): writer.write(np.full((100,100,3),i*10,dtype=np.uint8))
    writer.release()
    output=tmp_path/"annotated.mp4"; frames=list(video_frames(pipeline([]),str(input_path),str(output)))
    assert len(frames)==3 and frames[2][1]["timestamp_seconds"]==.2
    assert output.stat().st_size>0
    for suffix in ("json","jsonl","csv"):
        with PredictionWriter(tmp_path/f"predictions.{suffix}") as export:
            for _,record in frames: export.write(record)
    assert len(json.loads((tmp_path/"predictions.json").read_text()))==3
    assert "No person detected." in (tmp_path/"predictions.csv").read_text()
    with pytest.raises(ValueError): list(video_frames(pipeline([]),str(tmp_path/"missing.avi")))

def test_capture_release_on_early_close(monkeypatch):
    class Capture:
        released=False
        def isOpened(self): return True
        def get(self,key): return 10
        def read(self): return True,np.zeros((100,100,3),dtype=np.uint8)
        def release(self): self.released=True
    capture=Capture(); monkeypatch.setattr(cv2,"VideoCapture",lambda source:capture)
    iterator=video_frames(pipeline([]),0)
    next(iterator); iterator.close(); assert capture.released
