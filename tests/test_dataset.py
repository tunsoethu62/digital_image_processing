import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml
from uniform_system.dataset import group_split,split_dataset,label_counts,validate_manifest,extract_frames
from uniform_system.core import validate_class_map

def row(i,p,s,v): return dict(image=f"images/{i}.png",label=f"labels/{i}.txt",person_ids=p,session_id=s,source_video=v)

def test_transitive_group_leakage_and_determinism():
    rows=[row(0,"a","s0","v0"),row(1,"a;b","s1","v1"),row(2,"b","s2","v2"),row(3,"c","s3","v2")]
    rows += [row(i,f"p{i}",f"s{i}",f"v{i}") for i in range(4,20)]
    result=group_split(rows,42); location={i:s for s,indices in result.items() for i in indices}
    assert len({location[i] for i in range(4)})==1
    assert len(location)==len(rows) and result==group_split(rows,42)
    for key in ("person_ids","session_id","source_video"):
        seen={}
        for i,r in enumerate(rows):
            for value in r[key].split(";"):
                if value:
                    assert value not in seen or seen[value]==location[i]
                    seen[value]=location[i]

@pytest.mark.parametrize("label",["7 .5 .5 .2 .2","0 .1 .1 .5 .5","0 nan .5 .2 .2","0 .5 .5 -1 .2","0 .5 .5 .2"])
def test_bad_annotations(tmp_path,label):
    p=tmp_path/"bad.txt"; p.write_text(label)
    with pytest.raises(ValueError): label_counts(p)

def test_actual_split_files_and_distribution(tmp_path):
    (tmp_path/"images").mkdir(); (tmp_path/"labels").mkdir(); rows=[]
    for i in range(20):
        # Synthetic images exist only as test fixtures, not as project training data.
        cv2.imwrite(str(tmp_path/f"images/{i}.png"),np.full((30,30,3),i,dtype=np.uint8))
        (tmp_path/f"labels/{i}.txt").write_text("0 .5 .5 .5 .5\n")
        rows.append(row(i,f"p{i}",f"s{i}",f"v{i}"))
    manifest=tmp_path/"manifest.csv"
    with manifest.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    output=tmp_path/"split"; report=split_dataset(manifest,output)
    assert report["split_counts"]==dict(train=14,val=3,test=3)
    validate_class_map(yaml.safe_load((output/"dataset.yaml").read_text())["names"])
    assert validate_manifest(output/"manifest.csv")["class_counts"]["white_uniform_shirt"]==20
    with pytest.raises(ValueError): split_dataset(manifest,output)

def test_invalid_manifest_and_empty_labels(tmp_path):
    p=tmp_path/"empty.txt"; p.write_text(""); assert label_counts(p)=={}
    manifest=tmp_path/"bad.csv"; manifest.write_text("image,label\nx,y\n")
    with pytest.raises(ValueError): validate_manifest(manifest)

def test_frame_extraction_is_grouped_and_creates_no_labels(tmp_path):
    video=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(100,100))
    if not writer.isOpened(): pytest.skip('MJPG unavailable')
    for i in range(10): writer.write(np.full((100,100,3),i,dtype=np.uint8))
    writer.release()
    output=tmp_path/'frames'
    report=extract_frames(video,output,.3,'session','p1;p2')
    assert report['extracted_frames']==4
    with (output/'manifest.csv').open() as f: rows=list(csv.DictReader(f))
    assert all(r['session_id']=='session' and r['person_ids']=='p1;p2' for r in rows)
    assert len({r['source_video'] for r in rows})==1
    assert not (output/'labels').exists()
