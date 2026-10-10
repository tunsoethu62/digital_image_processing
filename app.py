"""Local demo. OpenCV webcams belong to the Python host, not a remote browser."""
import io
import json
import tempfile
import time
import yaml
from pathlib import Path
from dataclasses import asdict
import cv2
import numpy as np
import streamlit as st
from uniform_system.core import Rules, CLASS_NAMES
from uniform_system.pipeline import UniformPipeline, video_frames
from uniform_system.exports import PredictionWriter

st.set_page_config(page_title="Thai University Student Uniform Detection",layout="wide")
st.title("Thai University Student Uniform Detection System")
st.caption("Visible clothing assessment • Proposed project rules • KMUTNB pilot concept")
st.info("Requires trained seven-class clothing weights. No face recognition, gender inference, student identification or automatic penalties. A white shirt does not establish official uniform provenance.")
with st.sidebar:
    weights=st.text_input("Custom clothing weights path","runs/train/weights/best.pt")
    person_weights=st.text_input("Person detector path/checkpoint","yolo11n.pt")
    rules_path=st.text_input("Rules YAML","configs/rules.yaml")
    device=st.selectbox("Device",["auto","cpu","0"])
    try:
        defaults=Rules.load(rules_path)
    except (ValueError,OSError,TypeError,yaml.YAMLError) as exc:
        st.error(f"Invalid rules: {exc}")
        defaults=Rules()
    belt=st.checkbox("Require visible uniform belt",value=defaults.require_belt)
    emblem=st.checkbox("Require visible university emblem",value=defaults.require_emblem)
    override=st.checkbox("Override per-class confidence settings")
    thresholds={}
    if override:
        for name in CLASS_NAMES:
            thresholds[name]=st.slider(name,.01,1.,float(defaults.thresholds[name]),.01)
    st.caption("Defaults are demonstration thresholds, not validated optima.")
mode=st.radio("Input",["Image","Video","Webcam"],horizontal=True)
if mode=="Webcam":
    st.warning("Webcam index opens a camera on the computer running Python. It does not open a remote browser camera.")
    source=st.number_input("Python host webcam index",0,10,0)
    limit=st.number_input("Capture at most this many frames",1,3000,100)
    upload=None
else:
    upload=st.file_uploader("Upload image" if mode=="Image" else "Upload video",type=["jpg","jpeg","png","bmp"] if mode=="Image" else ["mp4","avi","mov","mkv"])
    limit=st.number_input("Video frame limit (0 = all)",0,100000,500) if mode=="Video" else 1
st.caption("Green: Compliant · Red: Non-compliant · Amber: Uncertain. Missing detection is not confirmed absence.")
start=st.button("Process",type="primary")
if start:
    if mode!="Webcam" and upload is None:
        st.error("Choose an input file.")
    else:
        # Each run has a finite limit. A rerun/stop interrupts this script; finally closes
        # the iterator. All outputs use temporary directories and disappear after export.
        iterator=None
        try:
            rules=Rules.load(rules_path)
            settings=asdict(rules); settings.update(require_belt=belt,require_emblem=emblem)
            if override:
                settings["thresholds"]=thresholds
                settings["candidate_confidence"]=min(settings["candidate_confidence"],min(thresholds.values()))
            rules=Rules(**settings)
            pipeline=UniformPipeline(weights,rules,person_weights,device)
            image_slot=st.empty(); summary_slot=st.empty(); started=time.perf_counter(); count=0
            with tempfile.TemporaryDirectory(prefix="uniform_demo_") as tmp:
                directory=Path(tmp)
                if mode=="Image":
                    raw=np.frombuffer(upload.getvalue(),dtype=np.uint8)
                    frame=cv2.imdecode(raw,cv2.IMREAD_COLOR) if raw.size else None
                    if frame is None: raise ValueError("Invalid or empty image")
                    annotated,record=pipeline.process(frame)
                    image_slot.image(annotated,channels="BGR")
                    summary_slot.json(record)
                    ok,encoded=cv2.imencode(".png",annotated)
                    if not ok: raise IOError("Image encode failed")
                    st.session_state["result"]=dict(media=encoded.tobytes(),media_name="annotated.png",mime="image/png",json=json.dumps([record],indent=2).encode(),csv=None)
                    with PredictionWriter(directory/"predictions.csv") as writer: writer.write(record)
                    st.session_state["result"]["csv"]=(directory/"predictions.csv").read_bytes()
                    count=1
                else:
                    if mode=="Webcam": actual_source=int(source)
                    else:
                        input_path=directory/("input"+Path(upload.name).suffix); input_path.write_bytes(upload.getvalue()); actual_source=str(input_path)
                    out=directory/"annotated.mp4"; json_path=directory/"predictions.json"; csv_path=directory/"predictions.csv"
                    iterator=video_frames(pipeline,actual_source,str(out),int(limit))
                    with PredictionWriter(json_path) as jw, PredictionWriter(csv_path) as cw:
                        try:
                            for annotated,record in iterator:
                                jw.write(record); cw.write(record); count+=1
                                image_slot.image(annotated,channels="BGR")
                                summary_slot.json(record)
                        finally: iterator.close(); iterator=None
                    st.session_state["result"]=dict(media=out.read_bytes(),media_name="annotated.mp4",mime="video/mp4",json=json_path.read_bytes(),csv=csv_path.read_bytes())
                wall=time.perf_counter()-started
                st.success(f"{count} frame(s); {wall:.2f} seconds; end-to-end {count/wall:.2f} FPS (includes UI/export).")
        except (ValueError,OSError,RuntimeError,ImportError,TypeError,yaml.YAMLError) as exc:
            st.error(str(exc)); st.session_state.pop("result",None)
        finally:
            if iterator is not None: iterator.close()
if "result" in st.session_state:
    result=st.session_state["result"]
    st.download_button("Save annotated output",result["media"],result["media_name"],result["mime"])
    st.download_button("Export predictions JSON",result["json"],"predictions.json","application/json")
    st.download_button("Export predictions CSV",result["csv"],"predictions.csv","text/csv")
st.caption("For long recordings use the streaming CLI; this demo holds downloadable outputs in memory. Stop from Streamlit's running indicator or interrupt the server; camera/video resources close in finally blocks.")
