import pytest
from uniform_system.core import Rules,Detection,assess,validate_class_map,CLASS_NAMES,crop_to_original,association_issues,validate_evidence

def d(k,conf=.95,issues=None): return Detection(k,conf,(10,20,80,100),issues or [])

@pytest.mark.parametrize("ids,belt,emblem,status",[
    ([0,2],False,False,"Compliant"),([0,3],False,False,"Compliant"),
    ([0,2,5],True,False,"Compliant"),([1,2],False,False,"Non-compliant"),
    ([0,4],False,False,"Non-compliant"),([0],False,False,"Uncertain"),
    ([0,2],True,False,"Uncertain"),([1],False,False,"Non-compliant"),
    ([0,2],False,True,"Uncertain"),([0,2,6],False,True,"Compliant"),
    ([],False,False,"Uncertain"),([0,1,2],False,False,"Uncertain"),
    ([0,2,4],False,False,"Uncertain"),([0,2,3],False,False,"Uncertain"),
    ([0,1,4],False,False,"Non-compliant"),
])
def test_decisions(ids,belt,emblem,status):
    assert assess([d(k) for k in ids],Rules(require_belt=belt,require_emblem=emblem)).status==status

def test_low_confidence_and_disabled_accessory():
    result=assess([d(0,.2),d(2)],Rules())
    assert result.status=="Uncertain" and "LOW_CONFIDENCE" in result.reason_codes
    assert assess([d(0),d(2),d(5,.1,["AMBIGUOUS_PERSON_ASSOCIATION"])],Rules()).status=="Compliant"

def test_clear_violation_precedes_occlusion():
    result=assess([d(1),d(2,.99,["CROPPED_GARMENT"])],Rules())
    assert result.status=="Non-compliant" and result.reason_codes==["NON_UNIFORM_UPPER"]

def test_ambiguous_violation_is_not_confirmed():
    result=assess([d(1,.99,["AMBIGUOUS_PERSON_ASSOCIATION"])],Rules())
    assert result.status=="Uncertain" and "AMBIGUOUS_PERSON_ASSOCIATION" in result.reason_codes

def test_per_person_evidence_never_combined():
    assert assess([d(0)],Rules()).status=="Uncertain"
    assert assess([d(2)],Rules()).status=="Uncertain"

def test_crop_mapping():
    assert crop_to_original((20,40,80,160),(100,200,300,600),2)==(110,220,140,280)

def test_class_map_and_config():
    validate_class_map(dict(enumerate(CLASS_NAMES)))
    with pytest.raises(ValueError): validate_class_map({0:"person"})
    with pytest.raises(ValueError): validate_class_map(list(CLASS_NAMES[:6]))
    with pytest.raises(ValueError): Rules(require_belt="false")
    with pytest.raises(ValueError): Rules(candidate_confidence=.9)
    with pytest.raises(ValueError): Rules(imgsz=0)
    assert Rules.load("configs/rules.yaml").require_emblem is False

def test_conservative_ownership_and_location():
    persons=[(0,0,100,250),(50,0,150,250)]
    assert association_issues((60,40,90,100),0,persons,Rules())==["AMBIGUOUS_PERSON_ASSOCIATION"]
    assert association_issues((10,40,40,100),0,persons,Rules())==[]
    invalid=validate_evidence(Detection(5,.99,(5,5,20,15)),0,persons,Rules())
    assert "INVALID_GARMENT_LOCATION" in invalid.issues


def test_conflicting_low_confidence_candidate_is_not_silently_resolved():
    result=assess([d(0,.2),d(1,.99),d(2)],Rules())
    assert result.status=="Uncertain" and "CONFLICTING_DETECTIONS" in result.reason_codes
    # A separately clear lower violation still has precedence.
    assert assess([d(0,.2),d(1,.99),d(4)],Rules()).status=="Non-compliant"
