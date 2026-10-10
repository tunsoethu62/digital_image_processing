# Verification record

Executed in the supplied cloud workspace on 2026-10-10 (Asia/Bangkok). Repository was initially empty, with no existing source or repository instructions. Environment: Linux x86_64, Python 3.12.14, CPU; CUDA unavailable.

## Installed and verified

| Dependency | Version |
|---|---|
| Ultralytics | 8.4.175 |
| PyTorch | 2.14.1 |
| torchvision | 0.29.1 |
| OpenCV (`opencv-python`) | 4.14.0.94 |
| NumPy | 2.5.3 |
| PyYAML | 6.0.3 |
| Streamlit | 1.65.0 |
| Pytest | 9.1.1 |

Direct versions are pinned in `requirements.txt`. The project is installed editable in `/workspace/uniform-venv`; this environment is outside the repository and is not required on another machine. No conflicting `opencv-python-headless` distribution remains. `pip check` reports no broken requirements. PyTorch runs on CPU here even though its pip distribution includes CUDA libraries.

## Checks performed

- `python -m pytest -q`: 43 passing tests. Synthetic images, detectors and mock checkpoints are confined to test fixtures in temporary directories. They are not shipped as dataset/model evidence.
- Rules tests cover all requested decisions, violation precedence, missing detection, conflicting upper/lower evidence, optional belt/emblem, low confidence and disabled accessory uncertainty.
- Geometry/pipeline tests cover per-person isolation, overlapping ownership, crop-to-original coordinates, empty scenes, invalid input and missing models.
- Real OpenCV test fixtures verify video writing/reading, annotated MP4 output, frame extraction, JSON/JSONL/CSV exports and capture release on early generator close.
- Dataset tests verify transitive person/session/video grouping, deterministic 70/15/15 on independent test groups, normalized label validation, copied split files, fixed names and distribution counts.
- Adapter tests verify training settings, checkpoint preprocessing metadata and dataset fingerprints, detection metrics fields and validator output-directory capture. These use mocks, not real training.
- Compliance evaluation tests verify one-to-one person matching, missing-person uncertainty and metric denominators.
- Streamlit AppTest starts the application, checks input selection and confirms the missing custom model message in webcam mode without attempting to open a camera.
- An actual headless Streamlit server started on 127.0.0.1:8501, with health endpoint HTTP 200. The verification server is stopped after the check.
- All 14 documented `uniform` command lines parse with the actual CLI, without pretending their user-supplied files exist. Dataset/rules class maps and JSON schema examples load consistently.
- Source compiles with `compileall`. Installed Ultralytics model predict/train/val signatures and metric attributes were inspected; the detection evaluator uses `on_val_end` to capture the actual validator output directory.

## Not performed / needed next

No user dataset, annotation files, person-level ground truth or custom trained weights were supplied. No actual training, full learned-model uniform inference, measured accuracy, threshold optimization, CLAHE comparison, GPU validation or real-camera benchmark was performed. A cloud CPU host with no attached webcam cannot demonstrate a live camera without hardware. Supply these resources through the documented workflow. The program does not fabricate predictions or numerical performance when they are missing.

Official documentation URLs were attempted but external retrieval failed in this environment. API inspection and executable adapter tests were used; the references remain linked in the README for access elsewhere.
