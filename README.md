# gotta detect 'em all: real-time pokemon detection

comparative study of object detection architectures applied to stylized video tracking. this project evaluates **yolov11s**, **rt-detr**, and **faster r-cnn** on a balanced dataset of 9 pokemon classes. the study analyzes the trade-offs between inference speed, localization quality, and classification precision to determine the optimal model for non-photorealistic domains.

![demo comparison](assets/demo.gif)
*(real-time tracking comparison)*

## performance benchmarks

models were evaluated on an **nvidia t4 gpu (16gb vram)** at a standardized resolution of **640x640**.

### 1. precision-recall comparison
the graph below highlights the significant performance gap. **yolov11s (green)** and **rt-detr (red)** maintain high precision across all recall levels, while **faster r-cnn (blue)** suffers a sharp drop, indicating a high rate of false positives.

![precision recall curve](results/__results___11_0.jpg)

### 2. quantitative metrics
yolov11s proves to be the superior architecture for real-time applications, achieving the highest f1-score (0.8943) with an inference time of just **10.99 ms**.

| model | precision | recall | f1 score | inference time |
| :--- | :--- | :--- | :--- | :--- |
| **yolov11s** | **0.9437** | **0.8499** | **0.8943** | **10.99 ms** |
| rt-detr | 0.8948 | 0.7905 | 0.8073 | 34.27 ms |
| faster r-cnn | 0.1836 | 0.7898 | 0.2980 | 88.29 ms |

## visual analysis

### confusion matrices: stability vs noise
comparing the confusion matrices reveals why faster r-cnn failed in precision.
* **yolov11s (left):** exhibits a sharp diagonal structure, indicating robust class separation and minimal background confusion.
* **faster r-cnn (right):** the top row is heavily populated, meaning the model frequently mistakes the **background** for a pokemon (hallucinations).

| yolov11s (clean) | faster r-cnn (noisy) |
| :---: | :---: |
| ![yolo cm](results/confusion_matrix_yolo.png) | ![frcnn cm](results/confusion_matrix_frcnn.png) |

### detection samples (yolov11s)
qualitative results from the best performing model. bounding boxes are tight and confidence scores are consistently high, even for stylized artwork.

![yolo detections](results/val_batch2_labels_yolo.jpg)

## methodology

### dataset & augmentation
we utilized a dataset of **9 classes** from roboflow universe. to mitigate initial class imbalance (e.g., 280 pikachu vs 22 eevee images), we implemented a synthetic augmentation pipeline:
* **classes:** gengar, greninja, snorlax, bulbasaur, charizard, charmander, eevee, pikachu, squirtle.
* **training set:** augmented to reach exactly **280 images per class**.

### architectures
* **yolov11s:** single-stage cnn with **frozen backbone (10 layers)** to prevent overfitting.
* **rt-detr:** transformer-based using self-attention for global context.
* **faster r-cnn:** two-stage detector fully fine-tuned from resnet50.

## usage

### installation
```bash
git clone [https://github.com/ricca200xx/Pokemon-Detection-Model-Comparison.git](https://github.com/ricca200xx/Pokemon-Detection-Model-Comparison.git)
cd Pokemon-Detection-Model-Comparison
pip install -r requirements.txt
