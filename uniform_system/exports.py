import csv
import json
from pathlib import Path

class PredictionWriter:
    """Incremental JSONL / person-level CSV for bounded-memory video processing."""
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.suffix not in (".json", ".jsonl", ".csv"):
            raise ValueError("Prediction export extension must be .json, .jsonl or .csv")
        self.file=self.path.open("w",newline=""); self.first=True
        self.csv=None
        if self.path.suffix==".json": self.file.write("[")
        if self.path.suffix==".csv":
            fields=["image_key","frame_number","timestamp_seconds","timestamp_source","source_fps","coordinate_space","width","height","person_index","temporary_track_id","person_confidence",
                    "person_box_xyxy","status","reason_codes","explanation","detections","latency_seconds","message"]
            self.csv=csv.DictWriter(self.file,fieldnames=fields); self.csv.writeheader()
    def write(self,record):
        if self.csv:
            for p in record["persons"] or [dict(person_index="",status="",message="No person detected.")]:
                row={k:record.get(k,"") for k in self.csv.fieldnames}
                row.update({k:v for k,v in p.items() if k in row})
                for k in ("person_box_xyxy","reason_codes","detections"):
                    row[k]=json.dumps(row[k])
                self.csv.writerow(row)
        elif self.path.suffix==".json":
            self.file.write(("," if not self.first else "")+json.dumps(record)); self.first=False
        else: self.file.write(json.dumps(record)+"\n")
    def close(self):
        if not self.file.closed:
            if self.path.suffix==".json": self.file.write("]\n")
            self.file.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
