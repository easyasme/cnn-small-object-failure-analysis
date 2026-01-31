# cnn-small-object-failure-analysis

Failure analysis of a CNN classifier on **small objects**, using a controlled condition dataset built from COCO validation images.

This project focuses on *why* CNNs struggle with small objects by isolating common visual degradation factors (resolution loss, motion blur, occlusion, and background clutter) and measuring how each factor affects performance across object sizes. Grad-CAM is used to analyze attention patterns in both correct and failed predictions.

---

## Problem Overview

CNNs are widely known to struggle with small objects, but the underlying reasons are often unclear.

Small-object recognition is challenging because:

- objects occupy only a small number of pixels
- repeated downsampling and pooling remove fine-grained details
- background clutter and contextual correlations can dominate predictions

To investigate these factors, this project uses a **controlled condition dataset**, where the same object instance is evaluated under different visual conditions while keeping object identity fixed.

---

## Dataset and Sampling (Step 1)

### Source
- COCO val2017

### Categories used
- person, car, truck, bus, bicycle, motorcycle

### Size grouping

Objects are grouped using the bounding-box area ratio(size thresholds defined in this project).

- **small**: bbox_area / img_area < 0.01
- **medium**: < 0.05
- **large**: >= 0.05

**Balanced Sampling**
- 200 samples are selected for each size group (small / medium / large)
- `person` category capped to reduce dominance

**Final Dataset**
- 600 object instances
- balanced across object sizes and categories

**Output**
- `data/samples_step1_person_limited.csv`

Each row includes the image path, bounding box, category label, size group, and area ratio.

---

## Controlled Condition Generation (Step 2)

For each object instance, six image conditions are generated:

- `orig_full`: original image
- `orig_crop`: cropped around bounding box with margin
- `downscale`: downscale + upsample (resolution loss)
- `motion_blur`: directional blur
- `occlusion`: partial occlusion inside bounding box
- `bg_blur`: blurred background with sharp object region

This step is **not data augmentation for training**.
It is designed for **controlled evaluation**, where only one visual factor changes at a time.

---

## Key Challenges and Design Decisions

### Challenge 1: Isolating failure causes
CNN failures on small objects are confused by multiple factors in standard datasets.

**Decision**
- Fix object identity and vary only one visual condition per sample.

---

### Challenge 2: Strong category imbalance in COCO
The `person` category dominates the dataset and can bias results.

**Decision**
- Cap the number of person samples per size group to improve balance.

---

## Challenge 3: Interpretability vs performance
High-capacity models can hide failure mechanisms.

**Decision**
- Use a simple, interpretable baseline (ResNet18 with frozen backbone).

---

## Baseline Model and Evaluation (Step 3)

**Model**
- ResNet18 (ImageNet pretrained)

**Training Setup**
- Backbone frozen
- Train only final fully connected layer
- 6-class classification

**Input**
- Per-class stratified 80/20 train/validation

**Training**
- Epochs: 5
- Optimizer: Adam (lr = 1e-3)

Setup prioritizes interpretability over absolute performance.

---

## Results

**Key Observations**
- Accuracy consistently increases with object size: small < medium < large
- Small objects underperform even in the original condition
- Cropping improves small-object accuracy, highlighting background clutter effects
- Downscaling and motion blur cause the largest accuracy drops for small objects
- Background blur benefits large objects but does not reliably fix small-object failures

**Saved Outputs**
- `data/results_conditions.csv`
- `data/step3_checkpoint.pth`

---

## Grad-CAM Failure Analysis (Step 4)

Grad-CAM is applied to the final convolutional block (layer4) of ResNet18.

**Observed Patterns**
- Correct predictions focus on object regions, even for small objects
- Failed predictions show diffuse or background-focused attention
- Small-object failures often rely on contextual cues rather than object features

This explains why small objects are especially vulnerable in cluttered scenes.

---

## Takeaways

- CNN failures on small objects are systematic, not random
- When objects are small, resolution loss(loss of detail) and background clutter hurt performance the most
- Focusing the model on the object itself by cropping greatly improves small-object accuracy
- Grad-CAM shows that when object features are weak, the model relies too much on background context

---

## Future Work
- Evaluate detector-style backbones for multi-scale features
- Test lightweight preprocessing methods like sharpening
- Extend experiments to more categories and larger datasets.
