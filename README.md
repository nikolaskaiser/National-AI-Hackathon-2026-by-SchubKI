# National AI Hackathon 2026 — SchubKI

Team workspace for the SchubKI hackathon.

## Challenge: Camera-Based Localization on a Race Track

**Partner:** Constructor Labs

Build a system that estimates a full-size autonomous race car's **position and heading on the Yas Marina circuit** from onboard camera streams. The intended use is supporting racing-line following at speed.

### Constraints

- Camera streams are the localization input.
- No GNSS or LiDAR.
- The system should remain robust at high speed and under motion blur, changing light, and low-texture asphalt.

### Known data

The challenge describes datasets collected from an autonomous race car on the F1 Yas Marina circuit. Dataset access, formats, camera details, coordinate frame, ground truth, and evaluation protocol still need to be confirmed.

## Initial plan

1. Inspect the dataset and document its camera streams, calibration, timestamps, coordinate conventions, and available labels.
2. Establish a reproducible baseline and offline evaluation before choosing a more complex method.
3. Compare localization quality and failure cases across lighting, blur, speed, and low-texture sections.
4. Keep the runtime input camera-only; use any ground truth only for training or evaluation if the challenge rules allow it.

**Starting hypothesis (to validate):** combine visual motion estimates with circuit-specific visual landmarks or map matching, then track a consistent position-and-heading estimate over time. The final approach depends on the data and evaluation rules.

## Open questions

- How do we access and load the dataset?
- Are camera calibration and synchronized ground-truth poses provided?
- What coordinate frame and output format are required?
- What are the evaluation metric, runtime constraints, and submission requirements?
- Are there rules about external data, pretrained models, or training resources?

## Team

- Nikolaskaiser (@nikolaskaiser)
- Teammate: GitHub username pending
