# Thai University Student Uniform Detection System

A runnable Digital Image Processing student project: a separate pretrained YOLO locates people, a **custom trained seven-class YOLO** detects clothing on each person's crop, and a configurable rules engine assesses visible evidence. Images, prerecorded videos and a webcam on the Python host are supported, with annotated outputs and JSON/CSV predictions.

**No custom weights or dataset are included.** The code and synthetic unit tests are complete; uniform accuracy, training results and preprocessing improvements have not been measured. A general pretrained YOLO is only a person detector / training initialization, never a completed uniform model. Missing custom weights produce a clear error and no invented clothing predictions.

The intended pilot is a controlled KMUTNB university/laboratory setting. Rules are **proposed project rules, not verified official KMUTNB regulations**. The system performs no face recognition, student identification, gender inference or automatic penalties. A visible white collared shirt does not prove official university provenance.

## Install

Python 3.11–3.13 is supported; development verification used Python 3.12. Run from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
uniform environment
```

The exact tested direct dependency versions are in `requirements.txt`; `docs/verification.md` records this environment. PyTorch/torchvision are pinned as a compatible pair; the regular pip build can install substantial CUDA libraries even on a CPU machine. CPU execution is always supported. For a smaller CPU-only install, install the same torch/torchvision versions from PyTorch's official CPU wheel index first if available for your platform, then install requirements. Use an official CUDA build matching your hardware when running on a GPU. `--device auto` selects CUDA device 0 if available, otherwise CPU; `--device cpu` forces CPU. No system OpenCV display window is required.

Dependencies in `pyproject.toml` provide compatible major-version ranges for development. Use requirements pins to reproduce the verified setup. A checkpoint may need a version matching its training environment. First use of `yolo11n.pt` requires internet access to download the pretrained checkpoint; supply a local checkpoint path offline.

## Files

```text
app.py                          local Streamlit interface
uniform_system/
  core.py                       fixed classes, rules and geometry
  pipeline.py                   people -> crops -> clothing -> decisions
  cli.py                        runnable commands
  dataset.py                    frame extraction, validation, grouped split
  model_tools.py                training and detection evaluation
  evaluation.py                 person compliance and timing metrics
  exports.py                    streaming JSON / JSONL / CSV
configs/
  rules.yaml                    proposed configurable rules
  dataset.yaml                  seven-class dataset template
  manifest.example.csv          schema example; referenced files not shipped
docs/
  annotation_guide.md            definitions, ambiguity and labeling workflow
  evaluation.md                 evaluation / threshold / CLAHE protocol
  model_metadata.example.json   external trained-model metadata template
  compliance_ground_truth.example.jsonl
  verification.md               actual checks and unresolved resources
tests/                          weight-free unit and integration tests
requirements.txt                exact direct dependency versions
```

## Dataset preparation and annotation

Planning target: about 1,000–2,000 consented, annotated person-crop images with variation in people, lighting, backgrounds, camera distance/angle, compliant and clearly violating outfits, occlusion and partial views. This is a target, not an existing dataset or a performance guarantee.

```bash
uniform extract recordings/session_A.mp4 data/session_A --interval 1.0 --session session_A --people 'P01;P02'
```

Extraction writes real frames plus a manifest; **it does not create labels**. Crop individual target people for clothing training (matching inference), annotate with CVAT or Label Studio, export YOLO boxes, and update image/label paths in the manifest for each crop. Keep every crop's original session, video and all known participant group IDs. Review the [annotation guide](docs/annotation_guide.md) before labeling. Unknown participants should remain linked through their session/video; avoid claiming person-disjoint splits when person groups are unavailable. Never force ambiguous garments into classes 1 or 4. Do not label an occluding jacket as a shirt violation.

Combine reviewed manifests into `data/raw/manifest.csv`. Paths are relative to the manifest directory. Columns: `image,label,person_ids,session_id,source_video`; multiple pseudonymous consent group IDs use `;`. Source-video/session values must be consistent across rows. Empty source video is permitted for photographs; session is mandatory. These private dataset IDs are never runtime identities.

```bash
uniform validate data/raw/manifest.csv
uniform split data/raw/manifest.csv data/split --seed 42
uniform validate data/split/manifest.csv
```

Validation checks decoding, box geometry, fixed IDs and class distribution. A reviewed negative image may have an empty label file, but a missing label file is an error. Split output must be empty. Splitting unions shared person, session and video groups transitively; exact duplicate files across splits are rejected. Aim for 70/15/15; large connected groups can prevent exact percentages or create empty splits. Collect more independent groups if needed. `data/split/distribution.json` reports actual counts, and `data/split/dataset.yaml` uses absolute paths. Do not put augmented training copies into validation/test. The seven names remain present when the optional emblem is unused. The configuration under `configs/` is a template; its data directories intentionally do not exist yet.

## Train

```bash
uniform train --data data/split/dataset.yaml --checkpoint yolo11n.pt --epochs 100 --batch 8 --imgsz 640 --seed 42 --device cpu --name baseline
```

Ultralytics fine-tunes pretrained detection weights and writes checkpoints, `results.csv`, logs and available plots under its actual run directory (normally `runs/baseline/`). Each saved `.pt` gets a `.metadata.json` recording preprocessing, classes, environment and dataset/settings. Class mapping is validated. Use a unique name for each run. All three split directories must contain data; exact class quality still needs a human label audit. For external genuinely trained weights, use [the metadata template](docs/model_metadata.example.json) as `best.metadata.json` next to `best.pt`, describing the actual training preprocessing. All seven checkpoint class names must match, even when accessories are disabled.

Optional experiment, using the same original cropped dataset and split:

```bash
uniform train --data data/split/dataset.yaml --checkpoint yolo11n.pt --preprocessing clahe --seed 42 --name clahe
```

The script creates a separate luminance-corrected dataset and trains/validates consistently. Set `preprocessing: clahe` in an alternate rules YAML to use these weights. Compare on validation before selecting the method; details are in [evaluation.md](docs/evaluation.md). Do not run CLAHE twice on already-corrected images. No whole-image Gaussian blur is enabled.

## Inference

After training, substitute the actual checkpoint path:

```bash
uniform infer image examples/photo.jpg --weights runs/baseline/weights/best.pt --output outputs/photo.png --predictions outputs/photo.json --image-key photo_A
uniform infer video recordings/demo.mp4 --weights runs/baseline/weights/best.pt --output outputs/demo.mp4 --predictions outputs/demo.jsonl
uniform infer webcam 0 --weights runs/baseline/weights/best.pt --output outputs/webcam.mp4 --predictions outputs/webcam.csv --max-frames 300
```

Input photos/recordings are user supplied; no example photo is fabricated. Stop video/webcam with Ctrl+C; captures and writers close in `finally`. A webcam index accesses the **computer running Python**, not a remote browser's camera. A cloud host without an attached camera cannot run the webcam demonstration. Invalid images, empty sources, missing weights, unavailable cameras and video open/read failures produce errors. OpenCV reports writer-open failure; some codec-level midstream write errors cannot be detected synchronously because `VideoWriter.write` has no success return. Inspect output files before relying on them. The default MP4 codec is `mp4v`; if browser playback is unsupported, download and play locally.

Each frame exports original-pixel boxes, class ID/name, detector confidence, rejected-evidence issues, local person index, status/reason codes/explanation, frame number and timestamp. JSON and JSONL preserve nested detections; CSV is one row per person with JSON-encoded box/detection fields and an empty-scene row when needed. No person yields **No person detected.**, with no outfit status. No compliance probability is invented. Track IDs are null; this version uses no tracking or temporal smoothing, so evidence cannot persist between people or frames.

## Local interface

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501 on the Python host (or use your cloud environment's port forwarding). Choose Image, Video or Webcam, weights and person model paths, confidence overrides, belt/emblem requirements and device. Results show green Compliant, red Non-compliant or amber Uncertain **with text**, component boxes/classes/confidences, full per-person JSON explanations and FPS. Download annotated image/video and JSON/CSV. The interface starts without weights, but processing requires them. Video/webcam have frame limits; Streamlit's stop control or server interruption closes resources. Large downloadable recordings use memory; use the streaming CLI for long recordings. This is a local demonstration, with no authentication or cloud services.

## Rules and image processing

`configs/rules.yaml` requires a white uniform-style shirt plus black formal trousers **or** black formal skirt, independent of gender. Belt and emblem are optional requirements. Thresholds are per class and are demonstration defaults, not measured optima. Missing detection always means unconfirmed, not absence. A confirmed valid replacement upper/lower garment gives Non-compliant even if another region is hidden. Conflicting positive/replacement evidence in a body region remains Uncertain unless an independent violation is clear. All enabled requirements with consistent evidence and no uncertainty give Compliant. Disabled accessory evidence cannot block Compliant. Enabled but unsupported model requirements fail the strict class-map check rather than being ignored.

OpenCV decodes BGR; Ultralytics handles RGB conversion, aspect-preserving letterboxing and normalization internally. Model boxes return in crop coordinates; integer crop origins map back to the full scene. No duplicate model resize/normalization is applied. Crop-border, small-region and rough body-location checks conservatively reject questionable evidence; any garment substantially overlapping another detected person's box is ambiguous. Bounding-box overlap cannot establish true silhouette ownership, hidden items or absence. Missed people, jackets incorrectly classified by the model, blurry yet confident predictions and subtle casual lookalikes remain risks. This version has no validated blur/absence/occlusion estimator. Demonstrate and evaluate these challenges before using decisions beyond a student project.

## Evaluation

```bash
uniform evaluate-detection --data data/split/dataset.yaml --weights runs/baseline/weights/best.pt --split val --name val_baseline --output outputs/detection_val.json
uniform evaluate-compliance data/reviews/val_truth.jsonl outputs/val_predictions.jsonl --output outputs/compliance_val.json
uniform thresholds data/reviews/val_truth.jsonl outputs/val_predictions.jsonl --class-name uniform_belt --values 0.4,0.5,0.6,0.7
uniform runtime outputs/val_predictions.jsonl
```

Produce validation prediction files using `uniform infer` on the reviewed validation scene videos; clothing crop labels are insufficient for full-scene compliance evaluation. Human review uses [the separate person-level format](docs/compliance_ground_truth.example.jsonl), frozen rules/visibility definitions and matching image keys/frame numbers. Detection and compliance metrics are separate; see [the evaluation protocol](docs/evaluation.md) for exact denominators, person-box matching, threshold selection, error analysis and baseline/CLAHE comparison. Runtime reports distinguish wall-clock FPS from processing-only latency and record hardware/resolution. After freezing settings, use `--split test` for detection and separately reviewed test scenes for final compliance. Do not tune on test.

## What you must supply

Consented recordings/images, reviewed seven-class crop labels and consistent grouping manifest, independent validation/test groups, person-level compliance ground truth, trained custom weights and truthful preprocessing metadata, pretrained person weights (download or local file), and an attached Python-host camera for live demonstration. Nothing here asserts official regulations or uniform detection accuracy.

## Technical references

- Ultralytics detection: https://docs.ultralytics.com/tasks/detect/
- YOLO detection dataset format: https://docs.ultralytics.com/datasets/detect/
- OpenCV: https://docs.opencv.org/4.x/
- DeepFashion background: https://liuziwei7.github.io/projects/DeepFashion.html
- DeepFashion2 background: https://arxiv.org/abs/1901.07973

Official documentation retrieval was attempted during implementation but the environment could not reach those URLs. API compatibility was checked against the installed Ultralytics/OpenCV packages instead. Clothing benchmark references do not establish Thai university uniform requirements.
