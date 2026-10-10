# Annotation guide

These seven labels describe visible appearance under proposed project rules. They do not establish official KMUTNB uniform provenance. Freeze this guide and rules before collecting labels; use consented participants and staged outfits. Never infer gender or identity from images.

## Label definitions (fixed IDs)

| ID | Name | Positive criteria |
|---|---|---|
| 0 | white_uniform_shirt | Clearly white, collared shirt/blouse of the agreed formal uniform-style cut. Label visible garment, not skin or collar alone. |
| 1 | non_uniform_upper_clothing | Clearly visible replacement primary upper garment violating the agreed white/collared/formal criteria, e.g. colored T-shirt or hoodie worn as the primary top in a staged known outfit. |
| 2 | black_uniform_trousers | Clearly black formal trousers; smooth formal fabric, tailored legs, no visible denim weave, rivets, faded patches or casual cargo pockets. |
| 3 | black_uniform_skirt | Clearly black skirt of the agreed formal style; annotate the skirt, not exposed legs. Do not choose this label from the wearer's appearance. |
| 4 | non_uniform_lower_clothing | Clear replacement lower garment violating configured black/formal criteria, e.g. colored trousers, visibly denim jeans, shorts or clearly casual skirt. |
| 5 | uniform_belt | Visible belt strap/buckle matching the agreed staged uniform-belt reference set; annotate only visible belt extent, not entire waist. |
| 6 | university_emblem | Clearly visible pin/emblem matching consented university reference examples. Optional extension; keep ID 6 even when unused. |

Camera appearance cannot reliably establish fabric composition, subtle black denim versus formal fabric, exact institution affiliation, concealed belt, pin authenticity, or exact skirt specifications. A white collared shirt can be a casual lookalike. If the distinction is uncertain, **do not assign a violation label**. Record the ambiguous region in review notes; omit its detection label. Such missing evidence becomes Uncertain at inference. Omission alone cannot train an explicit ambiguity detector; ambiguous lookalikes remain a documented model limitation.

## Image preparation and boxes

The clothing model receives person crops at inference. Prepare person-crop training images from the source recordings too, with all coordinates relative to the crop. Preserve original people/session/video groups in the manifest for each new crop. Include modest crop/background variation without transferring unrelated people across splits. Full-scene labels must be converted to crop coordinates before training. For crowded crops, annotate the target person's visible garments only, record others as a crowd/occlusion challenge, and manually audit; overlapping inference evidence is filtered conservatively.

Use CVAT or Label Studio with a YOLO object-detection export. Create the classes in the exact order above. Export plain text `class_id center_x center_y width height`, normalized to [0,1], one line per box. Preserve the seven-class names table even if the emblem has zero annotations. Check export order with `uniform validate` and manually review representative exports. Draw a tight rectangular box around the visible garment; no imagined hidden extension. Do not include a person's entire torso as a belt box. Do not draw a face box.

For partially visible garments, label only when enough defining features are visible to establish the class confidently. Clip boxes to image boundaries. A collar alone, a narrow black patch, or a blurred buckle does not justify a garment label. Record partial/hidden status in separate person-level review notes. A fully hidden garment has no box and must not be labeled absent or non-uniform.

A jacket, bag, scarf, folded arms or backpack covering a shirt is an **occluder**, not automatic class 1 evidence. Omit the covered underlying garment when its defining features cannot be seen. In particular, do not label a jacket as a replacement top merely because no shirt is visible. This version has no occluder class: training curation and cautious human review are necessary, and false violation risk must be measured on jackets. A separately clear violating lower garment can still determine Non-compliant.

For belts and emblems, require enough native pixels to distinguish the reference. Enlarge the viewer for placement, but do not treat interpolated pixels as evidence. If too small or blurry, omit and record unassessable. Keep negative/background person crops with genuinely reviewed empty label files; the tool never creates labels automatically. Images with a hidden garment can still have other valid garment labels.

## Review workflow

Two reviewers independently inspect difficult cases against the same frozen guide. Resolve disagreements, retain notes for ambiguous cases, and audit class 1/4 examples for actual replacement garments. Freeze rules, guide version, preprocessing and threshold settings for person-level compliance ground truth. Use pseudonymous consent group IDs only in the private dataset manifest, not runtime identities. Do not publish identifiable consented recordings as part of this repository.
