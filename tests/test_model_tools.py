"""No real training: mocks verify adapter semantics and on-disk run records."""
import json
from types import SimpleNamespace
import sys
import cv2
import numpy as np
import pytest
import yaml
from uniform_system.core import CLASS_NAMES,Rules
from uniform_system.model_tools import evaluate_detection,train
from uniform_system.dataset import resolve_dataset

def dataset(tmp_path):
    data=tmp_path/'data'
    for split in ('train','val','test'):
        image=data/'images'/split/'a.png'; image.parent.mkdir(parents=True)
        cv2.imwrite(str(image),np.zeros((50,50,3),dtype=np.uint8))
        label=data/'labels'/split/'a.txt'; label.parent.mkdir(parents=True);label.write_text('0 .5 .5 .5 .5\n')
    config=tmp_path/'dataset.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(data),train='images/train',val='images/val',test='images/test',names=dict(enumerate(CLASS_NAMES)))))
    return config

class Values(list):
    def tolist(self): return list(self)

def test_detection_metrics_adapter(tmp_path,monkeypatch):
    config=dataset(tmp_path); weights=tmp_path/'best.pt';weights.write_bytes(b'mock only')
    weights.with_suffix('.metadata.json').write_text(json.dumps(dict(preprocessing='baseline')))
    class Fake:
        def __init__(self,*args,**kwargs): self.names=dict(enumerate(CLASS_NAMES))
        def add_callback(self,event,callback): assert event=='on_val_end';self.callback=callback
        def val(self,**kwargs):
            self.callback(SimpleNamespace(save_dir=tmp_path/'run'))
            return SimpleNamespace(box=SimpleNamespace(ap_class_index=Values([0]),p=[.7],r=[.8],ap50=[.6],ap=[.4],map50=.6,map=.4))
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Fake))
    monkeypatch.setattr('uniform_system.model_tools.device_name',lambda d:'cpu')
    monkeypatch.setattr('uniform_system.model_tools.environment',lambda:{'fixture':True})
    args=SimpleNamespace(data=str(config),weights=str(weights),preprocessing='baseline',project=str(tmp_path/'runs'),name='validation',split='val',imgsz=640,device='cpu',output=str(tmp_path/'metrics.json'))
    result=evaluate_detection(args)
    assert result['artifacts']==str(tmp_path/'run')
    assert result['per_class']['white_uniform_shirt']['precision']==.7
    assert len(result['unrepresented_classes'])==6
    args.preprocessing='clahe'
    with pytest.raises(ValueError,match='mismatch'): evaluate_detection(args)

def test_train_records_checkpoints_and_fingerprints(tmp_path,monkeypatch):
    config=dataset(tmp_path);run=tmp_path/'runs'/'baseline';(run/'weights').mkdir(parents=True)
    (run/'weights'/'best.pt').write_bytes(b'mock only')
    class Fake:
        names=dict(enumerate(CLASS_NAMES))
        def __init__(self,*args,**kwargs): self.trainer=SimpleNamespace(save_dir=run)
        def train(self,**kwargs):
            assert kwargs['plots'] is True and kwargs['seed']==42
            assert kwargs['data'].endswith('dataset.yaml')
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Fake))
    monkeypatch.setattr('uniform_system.model_tools.device_name',lambda d:'cpu')
    monkeypatch.setattr('uniform_system.model_tools.environment',lambda:{'fixture':True})
    args=SimpleNamespace(data=str(config),preprocessing='baseline',project=str(tmp_path/'runs'),name='baseline',checkpoint='yolo11n.pt',epochs=1,batch=1,imgsz=64,seed=42,device='cpu')
    train(args)
    assert json.loads((run/'weights'/'best.metadata.json').read_text())['preprocessing']=='baseline'
    assert len(json.loads((run/'dataset_sha256.json').read_text()))==6
    with pytest.raises(ValueError): train(args)

def test_missing_labels_fail_training_dataset_validation(tmp_path):
    config=dataset(tmp_path)
    (tmp_path/'data'/'labels'/'train'/'a.txt').unlink()
    with pytest.raises(FileNotFoundError): resolve_dataset(config)
