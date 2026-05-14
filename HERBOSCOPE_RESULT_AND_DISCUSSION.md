# 4. RESULT AND DISCUSSION

This chapter presents the results of the proposed plant identification system developed in the Plant Model project. The system combines:

- A MobileNetV2-based classifier trained in `mobilenetmodel.ipynb`.
- A Salesforce BLIP transformer module for context-aware plant caption generation.
- A metadata enrichment pipeline based on curated plant descriptions.

The discussion includes both model-level evaluation and deployment-oriented optimization findings.

## 4.1 Model Evaluation

Model evaluation was carried out on train/validation/test splits from the `plant_dataset` directory. The classifier was trained using transfer learning, followed by fine-tuning of top MobileNetV2 layers. Performance was assessed using quantitative metrics and qualitative inspection of outputs.

### 4.1.1 Quantitative Analysis

#### Classification Metrics

From project experiments and report logs:

- Validation accuracy: approximately **95%**.
- Top-5 accuracy: approximately **98%**.
- GPU inference time (classification only): **300-500 ms/image**.

These values indicate that the model achieves high reliability for multiclass medicinal/plant species recognition under controlled test conditions.

#### Training Setup (from `mobilenetmodel.ipynb`)

- Backbone: `MobileNetV2(weights='imagenet', include_top=False)`.
- Input size: `224 x 224`.
- Head: `GlobalAveragePooling2D -> Dropout(0.4) -> Dense(256, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)`.
- Initial training: Adam (`lr = 1e-3`), early stopping, checkpointing.
- Fine-tuning: top layers unfrozen with Adam (`lr = 1e-5`), batch-norm layers frozen for stability.
- Saved outputs: `best_model.keras`, `best_model_finetune.keras`, `final_mobilenet_model.keras`.

#### Quantitative Interpretation

- Strong Top-1 and Top-5 performance confirms that transfer learning from ImageNet was effective for the plant domain.
- Top-5 significantly higher than Top-1 shows that ambiguous classes are still captured in candidate predictions.
- Remaining error is concentrated in species with visually similar leaves and limited samples.

### 4.1.2 Qualitative Analysis

Qualitative analysis focused on prediction behavior and generated captions.

#### Correct Prediction Patterns

The model performs best when:

- Leaf image is centered and clear.
- Lighting is balanced (no strong glare/shadow).
- Background clutter is low.
- Mature leaves with clear venation and edge features are visible.

#### Failure Patterns

Common difficult cases were:

- Similar morphology between species (shape, texture, vein pattern).
- Partial occlusions, damaged leaves, or disease artifacts.
- Perspective distortion from extreme camera angles.
- Complex backgrounds introducing non-leaf features.

#### BLIP Output Quality

Using Salesforce BLIP with prompt conditioning from plant metadata improved response quality:

- Captions became more plant-specific compared to generic image description.
- Outputs included useful semantic context (description/uses/family cues).
- Minor variability in phrasing occurred due to sampling-based decoding, but factual grounding improved with curated prompts.

## 4.2 Optimization Results

Optimization was evaluated with focus on deployment feasibility: memory footprint and response latency.

### 4.2.1 Model Size Reduction

#### Classifier Model

- Baseline classifier artifact (`final_mobilenet_model.keras`) is around **50 MB** (project-reported).
- This footprint is manageable for server deployment and can be further reduced for edge/mobile use through conversion and quantization.

#### BLIP Component

- BLIP is the dominant memory consumer in the pipeline.
- Project documentation indicates memory behavior around:
  - Full precision: about **3 GB** class footprint.
  - Quantized variant: about **2 GB**.

#### Trade-off Summary

- MobileNetV2 can be optimized aggressively with limited impact on accuracy.
- BLIP optimization reduces resource usage but may slightly reduce language richness.
- Prompt engineering and metadata grounding help preserve practical caption quality after optimization.

### 4.2.2 Inference Latency

Latency was measured at pipeline level and per-module level.

#### Observed Timing (GPU environment)

- MobileNetV2 classification: **300-500 ms**.
- BLIP caption generation: **500-1000 ms**.
- End-to-end response (API + model + formatting): typically **1.2-2.0 s**.

#### CPU-only Behavior

- Caption generation becomes the bottleneck.
- End-to-end time can increase to multi-second latency, reducing interactivity.

#### Optimization Insight

- For real-time UX, GPU serving or lightweight BLIP alternatives are preferred.
- If running on constrained hardware, mixed strategy is recommended:
  - Keep classifier local/on-edge.
  - Offload caption generation to server.

## 4.3 Discussion

### 4.3.1 Interpretation of Results

The results show that the combined architecture is effective for practical plant identification:

- Classification quality is high enough for academic and assisted field usage.
- Top-5 predictions reduce the risk of hard single-label failure.
- BLIP integration adds interpretability and user value beyond class name output.

Hence, the system does not only classify plants but also explains them in user-friendly language, which improves trust and usability.

### 4.3.2 Strengths of the System

Major strengths observed:

- Strong classification accuracy with transfer learning.
- Fast classifier inference suitable for interactive applications.
- Rich multimodal output using Salesforce BLIP (label + descriptive caption + metadata).
- Modular architecture: classifier, captioning, and metadata modules are independently improvable.
- Practical deployment flexibility across web backend and AI service components.

### 4.3.3 Challenges Encountered

Key challenges identified during development:

- Inter-class similarity between certain medicinal plants.
- Uneven sample distribution across classes.
- High compute and memory demand of BLIP models.
- Sensitivity to image acquisition conditions (lighting, blur, angle, background).
- Trade-off between caption diversity and deterministic reproducibility.

### Concluding Note

Overall, the project demonstrates that a MobileNetV2 + Salesforce BLIP pipeline is a strong solution for intelligent plant recognition. Future performance can be further improved by collecting more class-balanced data, adding hard-example mining for confusable species, and deploying optimized BLIP variants for lower-latency production systems.
