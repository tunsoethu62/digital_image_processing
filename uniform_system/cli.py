import argparse
import json
import sys
import time
import yaml
from dataclasses import asdict
from pathlib import Path
from .core import Rules, CLASS_NAMES, Detection, assess

def parser():
    p=argparse.ArgumentParser(description="Thai University Student Uniform Detection System")
    sub=p.add_subparsers(dest="command",required=True)
    infer=sub.add_parser("infer",help="Process image, video or Python-host webcam")
    infer.add_argument("kind",choices=["image","video","webcam"])
    infer.add_argument("source",help="File path or webcam index")
    infer.add_argument("--weights",required=True); infer.add_argument("--person-weights",default="yolo11n.pt")
    infer.add_argument("--rules",default="configs/rules.yaml"); infer.add_argument("--device",default="auto")
    infer.add_argument("--output",required=True); infer.add_argument("--predictions",required=True)
    infer.add_argument("--max-frames",type=int,default=0); infer.add_argument("--image-key",default="source")
    extract=sub.add_parser("extract",help="Extract real video frames; no fabricated labels")
    extract.add_argument("video"); extract.add_argument("output"); extract.add_argument("--interval",type=float,default=1)
    extract.add_argument("--session",required=True); extract.add_argument("--people",default="",help="Consented pseudonymous group IDs separated by ;")
    split=sub.add_parser("split"); split.add_argument("manifest"); split.add_argument("output"); split.add_argument("--seed",type=int,default=42)
    validate=sub.add_parser("validate"); validate.add_argument("manifest")
    for name in ("train","evaluate-detection"):
        q=sub.add_parser(name); q.add_argument("--data",default="data/split/dataset.yaml")
        q.add_argument("--imgsz",type=int,default=640); q.add_argument("--device",default="auto")
        q.add_argument("--project",default="runs"); q.add_argument("--name",default=name)
        q.add_argument("--preprocessing",choices=["baseline","clahe"],default="baseline")
        if name=="train":
            q.add_argument("--checkpoint",default="yolo11n.pt"); q.add_argument("--epochs",type=int,default=100)
            q.add_argument("--batch",type=int,default=8); q.add_argument("--seed",type=int,default=42)
        else:
            q.add_argument("--weights",required=True); q.add_argument("--split",choices=["val","test"],default="val")
            q.add_argument("--output",default="outputs/detection_metrics.json")
    ev=sub.add_parser("evaluate-compliance"); ev.add_argument("truth"); ev.add_argument("predictions")
    ev.add_argument("--match-iou",type=float,default=.5); ev.add_argument("--output",default="outputs/compliance_metrics.json")
    rt=sub.add_parser("runtime"); rt.add_argument("predictions"); rt.add_argument("--output",default="outputs/runtime.json")
    sweep=sub.add_parser("thresholds",help="Reassess cached validation candidates; never use test data to tune")
    sweep.add_argument("truth"); sweep.add_argument("predictions"); sweep.add_argument("--rules",default="configs/rules.yaml")
    sweep.add_argument("--values",default="0.4,0.5,0.6,0.7"); sweep.add_argument("--class-name",choices=CLASS_NAMES)
    sweep.add_argument("--output",default="outputs/threshold_sweep.json")
    sub.add_parser("environment")
    return p

def run(args):
    if args.command=="infer":
        from .pipeline import UniformPipeline, read_image, save_image, video_frames
        from .exports import PredictionWriter
        from .model_tools import environment
        if args.max_frames<0: raise ValueError("max-frames cannot be negative")
        rules=Rules.load(args.rules)
        pipeline=UniformPipeline(args.weights,rules,args.person_weights,args.device)
        start=time.perf_counter(); count=0
        with PredictionWriter(args.predictions) as writer:
            if args.kind=="image":
                image,record=pipeline.process(read_image(args.source)); record["image_key"]=args.image_key
                save_image(args.output,image); writer.write(record); count=1
            else:
                source=int(args.source) if args.kind=="webcam" else args.source
                iterator=video_frames(pipeline,source,args.output,args.max_frames)
                try:
                    for _,record in iterator:
                        record["image_key"]=args.image_key; writer.write(record); count+=1
                finally: iterator.close()
        wall=time.perf_counter()-start
        report=dict(frames=count,wall_seconds=wall,end_to_end_fps=count/wall,environment=environment(),
                    rules=asdict(rules),weights=str(args.weights),source=args.source,
                    note="Wall time includes decode, processing, annotation, writer and prediction export; model loading excluded.")
        Path(args.predictions+".run.json").write_text(json.dumps(report,indent=2)); return report
    if args.command in ("extract","split","validate"):
        from .dataset import extract_frames, split_dataset, validate_manifest
        if args.command=="extract": return extract_frames(args.video,args.output,args.interval,args.session,args.people)
        if args.command=="split": return split_dataset(args.manifest,args.output,args.seed)
        return validate_manifest(args.manifest)
    if args.command=="train":
        from .model_tools import train
        return train(args)
    if args.command=="evaluate-detection":
        from .model_tools import evaluate_detection
        return evaluate_detection(args)
    if args.command in ("evaluate-compliance","runtime","thresholds"):
        from .evaluation import load_records, compliance_metrics, runtime_report
        predictions=load_records(args.predictions)
        if args.command=="runtime": result=runtime_report(predictions)
        elif args.command=="evaluate-compliance": result=compliance_metrics(load_records(args.truth),predictions,args.match_iou)
        else:
            import copy
            truth=load_records(args.truth); original=Rules.load(args.rules); result=[]
            for value in map(float,args.values.split(",")):
                settings=asdict(original)
                settings["thresholds"]={k:(value if not args.class_name or k==args.class_name else v) for k,v in original.thresholds.items()}
                rules=Rules(**settings); changed=copy.deepcopy(predictions)
                for frame in changed:
                    for person in frame["persons"]:
                        ds=[Detection(d["class_id"],d["confidence"],tuple(d["box_xyxy"]),d.get("issues",[])) for d in person["detections"]]
                        person.update(asdict(assess(ds,rules)))
                result.append(dict(threshold=value,class_name=args.class_name or "all",metrics=compliance_metrics(truth,changed)))
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2))
        return result
    if args.command=="environment":
        from .model_tools import environment
        return environment()

def main():
    p=parser(); args=p.parse_args()
    try: print(json.dumps(run(args),indent=2))
    except KeyboardInterrupt: print("Stopped; video/camera resources released.",file=sys.stderr); return 130
    except (ValueError,OSError,RuntimeError,ImportError,TypeError,KeyError,yaml.YAMLError) as exc:
        print(f"Error: {exc}",file=sys.stderr); return 2
    return 0

if __name__=="__main__": sys.exit(main())
