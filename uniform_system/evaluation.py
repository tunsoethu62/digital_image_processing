import json
from pathlib import Path
from .core import STATUSES, iou

def load_records(path):
    text=Path(path).read_text()
    if not text.strip(): raise ValueError("Empty evaluation input")
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(s) for s in text.splitlines() if s.strip()]

def compliance_metrics(truth,predictions,match_iou=.5):
    """GT file uses image_key + frame_number + persons{person_key, box_xyxy,status}.
    Predictions use image_key (or single default source), original boxes and frame indices.
    Matching is one-to-one greedy descending IoU; unmatched GT defaults to Uncertain.
    Unmatched predicted people are reported separately, never silently scored as GT.
    """
    if not 0<match_iou<=1: raise ValueError("match_iou must be in (0,1]")
    pred={}
    for frame in predictions:
        key=(frame.get("image_key","source"),frame["frame_number"])
        if key in pred: raise ValueError("Duplicate prediction frame key")
        pred[key]=frame
    matrix=[[0]*3 for _ in range(3)]; unmatched_gt=0; unmatched_pred=0; seen=set()
    for frame in truth:
        key=(frame.get("image_key","source"),frame["frame_number"])
        if key in seen: raise ValueError("Duplicate ground truth frame key")
        seen.add(key)
        gt=frame["persons"]; pp=pred.get(key,{}).get("persons",[])
        if len({p["person_key"] for p in gt})!=len(gt): raise ValueError("Duplicate GT person keys")
        pairs=sorted([(iou(g["box_xyxy"],p["person_box_xyxy"]),i,j) for i,g in enumerate(gt) for j,p in enumerate(pp)],reverse=True)
        match={}; used=set()
        for score,i,j in pairs:
            if score>=match_iou and i not in match and j not in used: match[i]=j; used.add(j)
        unmatched_pred+=len(pp)-len(used)
        for i,g in enumerate(gt):
            if g["status"] not in STATUSES: raise ValueError("Unknown ground-truth status")
            predicted=pp[match[i]]["status"] if i in match else "Uncertain"
            if i not in match: unmatched_gt+=1
            if predicted not in STATUSES: raise ValueError("Unknown prediction status")
            matrix[STATUSES.index(g["status"])][STATUSES.index(predicted)]+=1
    total=sum(map(sum,matrix))
    if not total: raise ValueError("No ground truth people")
    f1=[]
    for i in range(3):
        denom=sum(matrix[i])+sum(row[i] for row in matrix)
        f1.append(2*matrix[i][i]/denom if denom else 0.0)
    uncertain=sum(row[2] for row in matrix); decided=total-uncertain
    compliant_truth=sum(matrix[0]); false_violations=matrix[0][1]
    extra_frames=sum(len(v["persons"]) for k,v in pred.items() if k not in seen)
    return dict(labels=list(STATUSES),confusion_matrix_rows_truth_columns_prediction=matrix,
                people=total,macro_f1=sum(f1)/3,accuracy=sum(matrix[i][i] for i in range(3))/total,
                uncertain_rate=uncertain/total,decision_coverage=decided/total,
                accuracy_among_decided=(matrix[0][0]+matrix[1][1])/decided if decided else None,
                false_violation_rate=false_violations/compliant_truth if compliant_truth else None,
                false_violation_numerator=false_violations,false_violation_denominator=compliant_truth,
                false_violation_definition="Predicted Non-compliant among ground-truth Compliant people",
                unmatched_ground_truth_people=unmatched_gt,unmatched_prediction_people=unmatched_pred,
                predictions_on_unlabeled_frames=extra_frames,match_iou=match_iou)

def runtime_report(records):
    import statistics
    times=[r["latency_seconds"] for r in records]
    import math
    if not times or any(not math.isfinite(t) or t<=0 for t in times):
        raise ValueError("Need positive finite frame timings")
    return dict(frames=len(times),processing_fps=len(times)/sum(times),
                mean_latency_seconds=statistics.mean(times),median_latency_seconds=statistics.median(times),
                resolution_counts={f'{r["width"]}x{r["height"]}':sum(t["width"]==r["width"] and t["height"]==r["height"] for t in records) for r in records},
                note="Processing timings include person/clothing inference, rules and drawing; exclude file decoding, export and UI. CLI also reports wall-clock FPS.")
