# gotta detect 'em all: real-time pokemon detection

comparative study of object detection architectures applied to stylized video tracking. this project evaluates **yolov11s**, **rt-detr**, and **faster r-cnn** on a balanced dataset of 9 pokemon classes. the study analyzes the trade-offs between inference speed, localization quality, and classification precision to determine the optimal model for non-photorealistic domains.

![demo comparison](assets/demo.gif)
*(real-time tracking comparison: yolov11s vs baselines)*

## performance benchmarks

models were evaluated on an **nvidia t4 gpu (16gb vram)** at a standardized resolution of **640x640**.

### 1. classification and speed efficiency
yolov11s proves to be the superior architecture for real-time applications, achieving the highest f1-score (0.8943) with an inference time of just **10.99 ms**. faster r-cnn serves as a baseline, showing high recall but suffering from extreme precision degradation due to background hallucinations.

| model | precision | recall | f1 score | inference time |
| :--- | :--- | :--- | :--- | :--- |
| **yolov11s** | **0.9437** | **0.8499** | **0.8943** | **10.99 ms** |
| rt-detr | 0.8948 | 0.7905 | 0.8073 | 34.27 ms |
| faster r-cnn | 0.1836 | 0.7898 | 0.2980 | 88.29 ms |

### 2. localization quality (iou & map)
while faster r-cnn fails in classification precision, its region proposal network (rpn) achieves the highest geometric accuracy on correctly identified objects (avg iou 0.7976), slightly outperforming the single-stage regressors.

| model | map 50-95 | avg iou (true positives) |
| :--- | :--- | :--- |
| **yolov11s** | **0.7871** | 0.7900 |
| rt-detr | 0.7387 | - |
| faster r-cnn | - | **0.7976** |

## visual analysis

### confusion matrices
* **yolov11s:** exhibits a sharp diagonal structure, indicating robust class separation and minimal confusion between similar shapes (e.g., bulbasaur vs charmander).
* **faster r-cnn:** the confusion matrix reveals heavy population in the top row (background class), confirming the model's tendency to generate false positives in empty space.

![confusion matrix comparison](results/confusion_matrix.png)
*(left: yolov11s showing clean predictions; right: faster r-cnn showing high background error rate)*

## methodology

### dataset & augmentation
we utilized a dataset of **9 classes** from roboflow universe. to mitigate initial class imbalance (e.g., 280 pikachu vs 22 eevee images), we implemented a synthetic augmentation pipeline:
* **classes:** gengar, greninja, snorlax, bulbasaur, charizard, charmander, eevee, pikachu, squirtle.
* **training set:** augmented to reach exactly **280 images per class**.
* **validation set:** augmented to reach exactly **90 images per class**.
* **transformations:** geometric (rotations, flips) and pixel-level (brightness, blur).

### architectures & training strategy
* **yolov11s (winner):** single-stage cnn.
    * *strategy:* **frozen backbone (first 10 layers)** to preserve domain-agnostic features and prevent overfitting on the stylized dataset.
* **rt-detr:** transformer-based with self-attention.
    * *strategy:* backbone partially frozen. uses global context to overcome locality constraints.
* **faster r-cnn (resnet50):** two-stage detector.
    * *strategy:* **full fine-tuning**. trained from scratch to adapt the deep feature extractors to the flat shading of pokemon characters, as pre-trained imagenet weights were insufficient for this domain gap.

## usage

### installation
```bash
git clone https://github.com/ricca200xx/Pokemon-Detection-Model-Comparison.git
cd Pokemon-Detection-Model-Comparison
pip install -r requirements.txt
