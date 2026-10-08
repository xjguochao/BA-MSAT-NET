# README: Relationship Between Processed Data and Figures

Paper: BA-MSAT-Net: A Multi-Scale Attention and Transformer for 3D Tooth Segmentation in CBCT Images
Manuscript ID: TRR-202609-02
Submitted:2026-09-26

---

## 1. Data Acquisition and Processing Pipeline

### 1.1 Raw Data
| Item | Specification |
|---|---|
| Total CBCT scans | 4,695 |
| Collection period | January 2004 – December 2025 |
| Acquisition sites | First Affiliated Hospital of Shihezi University; First Affiliated Hospital of Xinjiang Medical University |
| Scanner systems | Meiya® CBCT (Meiya Medical, Anhui, China); Dentium® CBCT (Dentium, Korea) |
| Export format | DICOM |
| Ethics approval | IRB No. KJ2024-295-02; KJ2025-038-01 |

### 1.2 Preprocessing Steps
| Step | Operation | Output |
|---|---|---|
| 1 | Anonymization and patient coding | De-identified DICOM series |
| 2 | Reorientation to midsagittal plane (MSP) | Standardized orientation |
| 3 | Resampling to isotropic voxel spacing | 0.4 × 0.4 × 0.4 mm³ |
| 4 | Patch partitioning | 160 × 160 × 160 voxel patches, infinite stride |
| 5 | Voxel-wise argmax inference → restore to original dimensions/orientation/origin/spacing/affine | Final segmentation masks |

### 1.3 Data Split
| Partition | Count | Percentage |
|---|---|---|
| Training | 3,287 | 70% |
| Validation | 939 | 20% |
| Test | 469 | 10% |

### 1.4 Segmentation Labels (4 classes)
1. **Tooth crown**
2. **Tooth root**
3. **Maxilla** (maxillary alveolar bone)
4. **Mandibular** (mandibular alveolar bone)

### 1.5 Regions of Interest (ROIs) for Boundary-Level Evaluation
| ROI | Definition |
|---|---|
| Apical root tip | Apical one-third of each root, measured from root apex along tooth-specific long axis |
| Root–bone interface | Portion of root surface within 1 mm of adjacent alveolar-bone surface |
| Interproximal alveolar bone | Alveolar bone between proximal root surfaces of two adjacent teeth, extending apically from alveolar crest to coronal one-third level of adjacent roots |

---

## 2. Figure–Data Mapping

### Fig. 1 — Architecture of the Proposed BA-MSAT-Net Framework
| Attribute | Detail |
|---|---|
| **Data dependency** | Conceptual / schematic; no experimental data |
| **Source** | Model design (residual 3D encoder–decoder + multi-scale spatial module + attention-gated skip fusion + Transformer self-attention at bottleneck + auxiliary boundary-prediction branch) |
| **Key components shown** | Residual block (Eq. 1), attention-gated skip fusion (Eqs. 2–3), scaled dot-product self-attention (Eq. 4), compound loss with boundary term (Eqs. 5–7) |

### Fig. 2 — Precision and Recall of V-Net, Mask-RCNN, and BA-MSAT-Net
| Attribute | Detail |
|---|---|
| **Data source** | Test set (n = 469), patient-level metrics |
| **Underlying table** | Tables 3–5 (class-specific DSC, Sensitivity, Precision, ASD, HD) |
| **Metrics plotted** | Precision (%) and Recall (= Sensitivity, %) per class and per model |
| **Classes** | Tooth crown, tooth root, maxilla, mandibular |
| **Models compared** | V-Net, Mask R-CNN, BA-MSAT-Net |

### Fig. 3 — Segmentation and Identification by BA-MSAT-Net, V-Net, and Mask-RCNN
| Attribute | Detail |
|---|---|
| **Data source** | Qualitative test-set samples (representative cases) |
| **Input** | Original CBCT slices (processed: MSP-reoriented, 0.4 mm isotropic) |
| **Overlay** | Predicted segmentation masks (4 classes) overlaid on CBCT slices |
| **Comparison** | Side-by-side: V-Net vs. BA-MSAT-Net vs. Mask R-CNN |
| **Regions emphasized** | Apical root tip, root–bone interface, maxilla/mandibular boundaries |
| **Related quantitative table** | Table 6 (boundary-level DSC, Boundary F1, HD, ASD) |

### Fig. 4 — Heatmap of Segmentation and Identification
| Attribute | Detail |
|---|---|
| **Data source** | Activation maps from the bottleneck / decoder of each model on test-set samples |
| **Panels** | (A) V-Net; (B) BA-MSAT-Net; (C) Mask R-CNN |
| **What it shows** | Spatial distribution of feature responses; BA-MSAT-Net shows concentrated, anatomically consistent activations over tooth structures with fewer background responses |
| **Related text** | Model Comparison section (lines 355–367) |

### Fig. 5 — Segmentation by BA-MSAT-Net Framework (3D Reconstructions)
| Attribute | Detail |
|---|---|
| **Data source** | Post-processed full-volume predictions on test-set cases (restored to original image dimensions after patch inference) |
| **Panels A–C** | Craniofacial 3D reconstructions from different viewing directions (tooth, maxilla, mandibular) |
| **Panels D–F** | Dental reconstructions showing crown–root morphology and dental arch continuity |
| **Processing link** | Argmax → restore original dimensions/orientation/origin/spacing/affine → surface rendering |
| **Related text** | Lines 363–367 (3D reconstruction, crown–root morphology, dental arch continuity) |

### Fig. 6 — Quantitative Ablation Analysis of Architectural Components
| Attribute | Detail |
|---|---|
| **Data source** | Ablation experiment on test set; 7 variants (see Table 9) |
| **Panel (A)** | Crown DSC, root DSC, and boundary F1-score across 7 ablation variants |
| **Panel (B)** | Boundary-region Hausdorff distance (HD) across variants (lower = better) |
| **Panel (C)** | Trade-off between dental-tissue DSC and number of trainable parameters |
| **Underlying table** | Table 9 |
| **Variants** | V-Net; +Multi-scale; +Transformer; +Attention gate; +Multi-scale+Transformer; MSAT-VNet; BA-MSAT-Net |

### Fig. 7 — Quantitative Ablation Comparison in Anatomically Challenging Regions
| Attribute | Detail |
|---|---|
| **Data source** | Boundary-level ablation results on the three ROIs |
| **Panel (A)** | Apical root tips |
| **Panel (B)** | Root–bone interface |
| **Panel (C)** | Interproximal alveolar bone |
| **Metrics** | DSC, Boundary F1-score, HD, ASD (per ROI, per variant) |
| **Underlying data** | Derived from Table 9 boundary metrics + Table 6 ROI-level evaluation |

---

## 3. Table–Data Mapping

### Table 1 — Contextual Comparison of Representative Published Methods
| Attribute | Detail |
|---|---|
| **Data type** | Literature review (no experimental data) |
| **Studies compared** | Cui et al. [2], Liu et al. [25], CISA-UNet [26], Trans-VNet [17], DentalSegmentator [3], This framework |
| **Columns** | Study, Task, Architecture, Dataset, Structures segmented, Reported metrics, Main difference from our study |

### Table 2 — Intra- and Inter-Examiner Reliability of Manual Segmentations
| Attribute | Detail |
|---|---|
| **Data source** | Manual reference annotations by 4 dental radiologists (3D Slicer v5.8.1) |
| **Intra-observer** | One annotator repeated segmentation after 2-week interval; two-way mixed-effects, absolute-agreement ICC |
| **Inter-observer** | Independent segmentations from all 4 experts; two-way random-effects, absolute-agreement ICC |
| **Classes** | Tooth crown, tooth root, maxilla, mandibular |
| **Metrics** | ICC with 95% CI |

### Table 3 — Class-Specific Segmentation Performance of BA-MSAT-Net
| Attribute | Detail |
|---|---|
| **Data source** | Test set (n = 469), patient-level means |
| **Classes** | Tooth crown, tooth root, maxilla, mandibular (plus two macro-averages) |
| **Metrics** | DSC (%), Sensitivity (%), ASD (mm), HD (mm), Precision (%) |
| **Key values** | Crown DSC 98.3%, Root DSC 94.1%, Maxilla DSC 94.3%, Mandibular DSC 94.7% |

### Table 4 — Class-Specific Segmentation Performance of V-Net
| Attribute | Detail |
|---|---|
| **Data source** | Same test set, same protocol as Table 3 |
| **Baseline model** | V-Net (3D volumetric segmentation) |
| **Metrics** | Identical to Table 3 |

### Table 5 — Class-Specific Segmentation Performance of Mask R-CNN
| Attribute | Detail |
|---|---|
| **Data source** | Same test set, same protocol as Table 3 |
| **Baseline model** | Mask R-CNN (instance-based) |
| **Metrics** | Identical to Table 3 |

### Table 6 — Boundary-Level Performance in Anatomical Structures
| Attribute | Detail |
|---|---|
| **Data source** | Test set, ROI-level evaluation (3 ROIs × 3 models) |
| **ROIs** | Apical root, Root–bone interface, Interproximal alveolar bone |
| **Models** | V-Net, Mask R-CNN, BA-MSAT-Net |
| **Metrics** | DSC (%), Boundary F1-score (%), HD (mm), ASD (mm) — all mean ± SD |
| **Note** | Surface-distance metrics computed in original patient space using physical distances (mm) |

### Table 7 — Robustness Analysis
| Attribute | Detail |
|---|---|
| **Data source** | Test set with simulated image degradation |
| **Conditions** | Original; Noise low; Noise high; Resolution downsampling 0.5×; Resolution downsampling 0.25× |
| **Models** | V-Net, Mask R-CNN, BA-MSAT-Net |
| **Metrics** | DSC (%), IoU (%), HD (mm), ASD (mm) — mean ± SD |

### Table 8 — Computational Efficiency Comparison
| Attribute | Detail |
|---|---|
| **Data source** | Benchmark measurements on identical hardware |
| **Input patch** | 160 × 160 × 160 voxels, batch size = 1 |
| **Models** | V-Net, Mask R-CNN, BA-MSAT-Net |
| **Metrics** | Parameters (M), FLOPs (G), Inference time (s), FPS, Convergence epoch, Final validation loss |
| **Note** | Inference time = neural-network forward + patch fusion; excludes data loading, ROI localization, post-processing, file writing |

### Table 9 — Ablation Analysis of Architectural Components
| Attribute | Detail |
|---|---|
| **Data source** | Test set, 7 variants with identical data split/augmentation/training/evaluation |
| **Variants** | V-Net; +Multi-scale; +Transformer; +Attention gate; +Multi-scale+Transformer; MSAT-VNet (no boundary loss); BA-MSAT-Net (full) |
| **Columns** | Multi-scale ✓, Transformer ✓, Attention gate ✓, Boundary loss ✓, Crown DSC, Root DSC, Boundary F1, HD (mm), Parameters |
| **Note** | Boundary F1 and HD averaged across apical-root, root-bone-interface, and alveolar-bone regions |

### Table S1 (Supplementary) — Demographic Characteristics of Participants
| Attribute | Detail |
|---|---|
| **Data source** | Full cohort (n = 4,695) |
| **Content** | Age, sex, and other demographic summaries |

---

## 4. Evaluation Metrics Definitions

| Metric | Formula / Definition | Unit |
|---|---|---|
| **DSC** | 2\|R∩G\| / (\|R\|+\|G\|), R=predicted, G=ground truth | % |
| **Sensitivity (Recall)** | TP / (TP + FN) | % |
| **Precision** | TP / (TP + FP) | % |
| **ASD** | Mean Euclidean distance between predicted and reference surfaces | mm |
| **HD** | 95th-percentile Hausdorff distance between surfaces | mm |
| **Boundary F1-score** | Surface distance + boundary overlap + anatomical continuity in low-contrast regions | % |
| **mAP@0.5** | Mean average precision at IoU threshold 0.5 (instance-level) | – |
| **IoU** | \|R∩G\| / \|R∪G\| | % |

---

## 5. Statistical Analysis Protocol

| Comparison | Test |
|---|---|
| Normality of paired differences | Shapiro–Wilk test |
| Two models, normal distribution | Two-sided paired *t*-test |
| Two models, non-normal | Wilcoxon signed-rank test |
| More than two models | ANOVA + post hoc pairwise with correction for multiple testing |
| Significance level | α = 0.05 |
| Reliability | ICC (intraclass correlation coefficient) with 95% CI |

---

## 6. Data Availability

- Zenodo:
- GitHub:
---

*This README maps every figure and table in the manuscript to its underlying processed data, evaluation protocol, and preprocessing stage. All numerical values are patient-level means on the independent test set nunless otherwise noted.*
