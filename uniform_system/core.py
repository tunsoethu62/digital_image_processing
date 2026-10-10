from dataclasses import dataclass, field
from pathlib import Path
import math
import yaml

CLASS_NAMES = (
    "white_uniform_shirt", "non_uniform_upper_clothing", "black_uniform_trousers",
    "black_uniform_skirt", "non_uniform_lower_clothing", "uniform_belt", "university_emblem",
)
STATUSES = ("Compliant", "Non-compliant", "Uncertain")

@dataclass
class Rules:
    require_belt: bool = False
    require_emblem: bool = False
    thresholds: dict = field(default_factory=lambda: dict(zip(CLASS_NAMES, [.5,.6,.5,.5,.6,.6,.65])))
    person_confidence: float = .4
    candidate_confidence: float = .1
    min_person_height: int = 96
    min_garment_side: int = 6
    association_margin: float = .2
    imgsz: int = 640
    preprocessing: str = "baseline"

    def __post_init__(self):
        if type(self.require_belt) is not bool or type(self.require_emblem) is not bool:
            raise ValueError("Accessory requirements must be booleans")
        if set(self.thresholds) != set(CLASS_NAMES):
            raise ValueError("Thresholds must name all seven fixed classes")
        for value in [*self.thresholds.values(), self.person_confidence, self.candidate_confidence,
                      self.association_margin]:
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Confidence, IoU and association values must be in (0, 1]")
        if self.candidate_confidence > min(self.thresholds.values()):
            raise ValueError("Candidate threshold must not exceed a class decision threshold")
        if any(type(v) is not int or v <= 0 for v in [self.min_person_height, self.min_garment_side, self.imgsz]):
            raise ValueError("Sizes must be positive integers")
        if self.preprocessing not in ("baseline", "clahe"):
            raise ValueError("preprocessing must be baseline or clahe")

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Rules file must be a YAML mapping")
        return cls(**data)

@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple
    issues: list = field(default_factory=list)

    def __post_init__(self):
        if self.class_id not in range(7):
            raise ValueError("Invalid clothing class ID")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid detection confidence")
        if len(self.box) != 4 or not all(math.isfinite(v) for v in self.box) or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            raise ValueError("Invalid bounding box")

    def export(self):
        return dict(class_id=self.class_id, class_name=CLASS_NAMES[self.class_id],
                    confidence=self.confidence, box_xyxy=list(self.box), issues=self.issues)

@dataclass
class Decision:
    status: str
    reason_codes: list
    explanation: str

EXPLANATIONS = {
    "ALL_REQUIREMENTS_CONFIRMED": "All enabled visible clothing requirements have consistent evidence.",
    "NON_UNIFORM_UPPER": "A confidently detected upper garment violates the proposed clothing rule.",
    "NON_UNIFORM_LOWER": "A confidently detected lower garment violates the proposed clothing rule.",
    "REQUIRED_COMPONENT_UNCONFIRMED": "A required component is unconfirmed; missing detection does not prove absence.",
    "LOW_CONFIDENCE": "Some clothing evidence is below its configured confidence threshold.",
    "CONFLICTING_DETECTIONS": "Conflicting clothing predictions cannot be reliably resolved.",
    "AMBIGUOUS_PERSON_ASSOCIATION": "Clothing ownership cannot be resolved conservatively.",
    "SMALL_REGION": "A person or garment region is too small for reliable assessment.",
    "INVALID_GARMENT_LOCATION": "Clothing evidence lies outside its plausible body region.",
    "CROPPED_GARMENT": "A garment reaches the person crop boundary and may be incomplete.",
}

def assess(detections, rules):
    """Each invocation receives ONE person's evidence. No absence assessment is performed."""
    valid = [d for d in detections if not d.issues and
             d.confidence >= rules.thresholds[CLASS_NAMES[d.class_id]]]
    candidates = [d for d in detections if not d.issues and d.confidence >= rules.candidate_confidence]
    # A crop is already a single body region; positive and replacement labels in
    # the same region conflict even if their boxes differ. No confidence tie-break.
    upper_conflict = any(d.class_id == 0 for d in candidates) and any(d.class_id == 1 for d in candidates)
    lower_conflict = any(d.class_id in (2,3) for d in candidates) and any(d.class_id == 4 for d in candidates)
    lower_shape_conflict = any(d.class_id == 2 for d in candidates) and any(d.class_id == 3 for d in candidates)
    violations = []
    if not upper_conflict and any(d.class_id == 1 for d in valid):
        violations.append("NON_UNIFORM_UPPER")
    if not lower_conflict and any(d.class_id == 4 for d in valid):
        violations.append("NON_UNIFORM_LOWER")
    if violations:
        return decision("Non-compliant", violations)
    present = {d.class_id for d in valid}
    missing = (0 not in present or not present.intersection({2,3}) or
               (rules.require_belt and 5 not in present) or (rules.require_emblem and 6 not in present))
    # Disabled accessory evidence cannot affect the decision.
    relevant = {0,1,2,3,4} | ({5} if rules.require_belt else set()) | ({6} if rules.require_emblem else set())
    uncertain = []
    for d in detections:
        if d.class_id in relevant:
            uncertain.extend(d.issues)
            if d.confidence < rules.thresholds[CLASS_NAMES[d.class_id]]:
                uncertain.append("LOW_CONFIDENCE")
    if upper_conflict or lower_conflict or lower_shape_conflict:
        uncertain.append("CONFLICTING_DETECTIONS")
    if missing:
        uncertain.append("REQUIRED_COMPONENT_UNCONFIRMED")
    if uncertain:
        return decision("Uncertain", uncertain)
    return decision("Compliant", ["ALL_REQUIREMENTS_CONFIRMED"])

def decision(status, codes):
    codes = list(dict.fromkeys(codes))
    return Decision(status, codes, " ".join(EXPLANATIONS[c] for c in codes))

def validate_class_map(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    actual = {int(k): v for k,v in names.items()}
    if actual != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"Clothing class map mismatch: expected {dict(enumerate(CLASS_NAMES))}, got {actual}")

def intersection(a,b):
    return max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))

def area(b):
    return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def iou(a,b):
    hit=intersection(a,b)
    return hit / max(area(a)+area(b)-hit, 1e-9)

def crop_to_original(box, crop, scale=1.0):
    """scale = resized crop pixels / original crop pixels (YOLO handles its own resize)."""
    if scale <= 0:
        raise ValueError("Scale must be positive")
    return (box[0]/scale+crop[0], box[1]/scale+crop[1], box[2]/scale+crop[0], box[3]/scale+crop[1])

def association_issues(box, owner_index, persons, rules):
    owner=persons[owner_index]
    if intersection(box,owner) / max(area(box),1e-9) < .9:
        return ["AMBIGUOUS_PERSON_ASSOCIATION"]
    # Any substantial overlap with another person's box is ambiguous. A box-only
    # heuristic cannot establish ownership where silhouettes overlap.
    if any(i != owner_index and intersection(box,p)/max(area(box),1e-9) > rules.association_margin
           for i,p in enumerate(persons)):
        return ["AMBIGUOUS_PERSON_ASSOCIATION"]
    return []

def validate_evidence(d, owner_index, persons, rules):
    p=persons[owner_index]; w=p[2]-p[0]; h=p[3]-p[1]
    d.issues.extend(association_issues(d.box,owner_index,persons,rules))
    if h < rules.min_person_height or min(d.box[2]-d.box[0], d.box[3]-d.box[1]) < rules.min_garment_side:
        d.issues.append("SMALL_REGION")
    cy=((d.box[1]+d.box[3])/2-p[1])/h
    ranges={0:(.08,.68),1:(.08,.68),2:(.35,1),3:(.35,1),4:(.35,1),5:(.28,.78),6:(.08,.60)}
    if not ranges[d.class_id][0] <= cy <= ranges[d.class_id][1]:
        d.issues.append("INVALID_GARMENT_LOCATION")
    if min(d.box[0]-p[0],d.box[1]-p[1],p[2]-d.box[2],p[3]-d.box[3]) <= 1:
        d.issues.append("CROPPED_GARMENT")
    d.issues=list(dict.fromkeys(d.issues))
    return d
