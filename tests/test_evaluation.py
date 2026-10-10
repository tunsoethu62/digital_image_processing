import pytest
from uniform_system.evaluation import compliance_metrics

def gt(status,key="p"): return dict(person_key=key,box_xyxy=[0,0,100,200],status=status)
def pred(status): return dict(person_box_xyxy=[0,0,100,200],status=status)

def test_metrics_denominators_and_matching():
    truth=[dict(frame_number=i,persons=[gt(s)]) for i,s in enumerate(["Compliant","Compliant","Non-compliant","Uncertain"])]
    predictions=[dict(frame_number=i,persons=[pred(s)]) for i,s in enumerate(["Non-compliant","Uncertain","Non-compliant","Uncertain"])]
    m=compliance_metrics(truth,predictions)
    assert m["accuracy"]==.5 and m["decision_coverage"]==.5
    assert m["accuracy_among_decided"]==.5 and m["false_violation_rate"]==.5
    assert m["false_violation_denominator"]==2

def test_missing_person_is_uncertain_and_predictions_cannot_double_match():
    truth=[dict(frame_number=0,persons=[gt("Compliant","a"),gt("Compliant","b")])]
    m=compliance_metrics(truth,[dict(frame_number=0,persons=[pred("Compliant")])])
    assert m["unmatched_ground_truth_people"]==1 and m["uncertain_rate"]==.5
    assert m["accuracy"]==.5
    with pytest.raises(ValueError): compliance_metrics([],[])
