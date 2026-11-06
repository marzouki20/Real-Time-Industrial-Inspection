<div align="center">

# Edge Visual Inspection System — TinyML on ESP32-CAM
### Real-Time Edge AI + IoT Monitoring System

[![Arduino](https://img.shields.io/badge/Arduino-IDE%202.x-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Micro-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-660066?style=for-the-badge&logo=eclipse-mosquitto&logoColor=white)](https://mosquitto.org/)
[![Node-RED](https://img.shields.io/badge/Node--RED-Dashboard-8F0000?style=for-the-badge&logo=node-red&logoColor=white)](https://nodered.org/)
[![Edge Impulse](https://img.shields.io/badge/Edge%20Impulse-TinyML-4B0082?style=for-the-badge)](https://edgeimpulse.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Deploying production-grade AI on a Rs.500 microcontroller.**
> Real-time visual anomaly detection at the edge — no cloud, no compromise.

<br/>

![Status](https://img.shields.io/badge/Status-Live%20%26%20Working-brightgreen?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Anomaly%20Recall-94.8%25-blue?style=for-the-badge)
![RAM](https://img.shields.io/badge/RAM%20Usage-280KB-orange?style=for-the-badge)

<br/>

[![Watch Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://youtube.com/shorts/fE8pEs_nPZ4)

</div>

---

## Problem Statement

Industrial motors and rotating equipment are critical assets in manufacturing plants, power stations, and processing facilities. Undetected anomalies — such as surface defects, misalignment, foreign object intrusion, or abnormal wear — lead to unplanned downtime, costly repairs, and serious safety hazards.

Current solutions fail in different ways:

| Approach | Problem |
|----------|---------|
| Manual Inspection | Slow, inconsistent, error-prone, dangerous |
| Cloud Vision AI | Expensive, high latency, internet-dependent, privacy risk |

Neither works in resource-constrained or remote environments.

This project deploys a quantized **MobileNetV2 model directly on an ESP32-CAM** — 520KB RAM, under Rs.500 — achieving **94.8% anomaly recall** with zero cloud dependency, over a secured MQTT pipeline feeding a live IoT dashboard.

---

## System Architecture

```
+-------------------------------+
|         ESP32-CAM             |
|  OV2640 Camera                |
|  MobileNetV2 int8 Inference   |
|  MQTT Publish                 |
+---------------+---------------+
                |
         WiFi (Secured MQTT)
                |
+---------------v---------------+
|   Mosquitto MQTT Broker       |
|   Auth Required               |
|   No Anonymous Connections    |
+---------------+---------------+
                |
+---------------v---------------+
|   Node-RED IoT Dashboard      |
|   Confidence Gauge            |
|   Motor Status                |
|   bcrypt Auth Protected       |
+-------------------------------+
```

---

## Model Performance

<div align="center">

| Metric | Value |
|--------|-------|
| Overall Accuracy | **87.8%** |
| Anomaly Recall | **94.8%** |
| Normal Recall | **78.4%** |
| F1 Score (Anomaly) | **0.90** |
| Model Architecture | MobileNetV2 96x96 (int8 quantized) |
| Inference Time | ~643ms / frame |
| RAM Footprint | ~280KB / 520KB |
| Model Size | < 1MB |

</div>

> In industrial safety, missing a real defect is far more dangerous than a false alarm.
> We optimized for anomaly recall — 94.8% means almost nothing slips through undetected.

---

## Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| Microcontroller | ESP32-CAM (AI Thinker) with OV2640 |
| Programmer | CP2102 USB-TTL |
| ML Framework | TensorFlow Lite Micro |
| Training Pipeline | Edge Impulse Studio |
| Firmware IDE | Arduino IDE 2.x |
| IoT Protocol | MQTT (PubSubClient) |
| Message Broker | Mosquitto |
| Dashboard | Node-RED |
| OS | Ubuntu 24 (WSL2) |
| Auth | bcrypt + MQTT Credentials |
| Dataset | Kaggle Casting Product QC (7,348 images) |

</div>

---

## Security Architecture

```
+--------------------------------------------------+
|                  Security Layers                 |
+--------------------------------------------------+
|  MQTT username + password authentication         |
|  Anonymous connections disabled                  |
|  Broker bound to 0.0.0.0 (all interfaces)        |
|  Node-RED editor protected with bcrypt hash      |
|  Credentials isolated in separate config files   |
+--------------------------------------------------+
|  Planned: mTLS mutual certificate authentication |
+--------------------------------------------------+
```

---

## Setup Instructions

### Complete training workflow in Google Colab

The dataset used in this project is the extracted archive located at:

```
archive/casting_data/casting_data/
├── train/def_front
├── train/ok_front
├── test/def_front
└── test/ok_front
```

The verified split contains 6,633 training images and 715 test images
(7,348 JPEG images total). `def_front` means defective casting and `ok_front`
means acceptable casting.

#### 1. Prepare the archive

Compress the project folder as a ZIP file. The ZIP must contain the
`archive/casting_data/casting_data` directory. Open
[Google Colab](https://colab.research.google.com), create a notebook, and
select **Runtime > Change runtime type > T4 GPU**.

#### 2. Upload and extract the project

Run this Colab cell:

```python
from google.colab import files
import zipfile

uploaded = files.upload()
zip_name = next(iter(uploaded))

with zipfile.ZipFile(zip_name, "r") as archive:
    archive.extractall("/content")
```

Find the extracted dataset:

```python
from pathlib import Path

dataset_roots = []
for train_dir in Path("/content").rglob("train"):
    candidate = train_dir.parent
    if (
        (train_dir / "def_front").is_dir()
        and (train_dir / "ok_front").is_dir()
        and (candidate / "test").is_dir()
    ):
        dataset_roots.append(candidate)

if not dataset_roots:
    raise FileNotFoundError("Dataset train/test not found")

data_dir = dataset_roots[0]
print(data_dir)
```

#### 3. Install the Colab dependencies

```python
!pip install -q "tensorflow>=2.15,<2.21" "numpy<2" scikit-learn seaborn
```

#### 4. Train and export the model

The repository script trains MobileNetV2 with 96×96 images, uses class
weights to compensate for the class imbalance, evaluates on the existing test
split, and exports an INT8 TensorFlow Lite model:

```python
%cd /content/Edge-Visual-Inspection-ESP32-main/Edge-Visual-Inspection-ESP32-main
!python training/train.py --data-dir "{data_dir}" --epochs 15 --batch-size 32
```

If the extracted folder has another name, run the script from its actual
location or use the absolute path in `--data-dir`.

The generated files are:

```
model/export/
├── best.keras
├── labels.txt
├── metrics.json
├── model.keras
└── model_int8.tflite
```

#### 5. Display all model statistics

The command above saves the final metrics and model files. For interactive
curves and detailed statistics, run the training code as notebook cells
(rather than through `!python`) so that `history`, `model`, `test_ds`, and
`class_names` remain available. Then run the statistics cell below:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

y_true = []
y_probability = []
for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)
    y_probability.extend(probabilities)
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_probability = np.array(y_probability)
y_pred = np.argmax(y_probability, axis=1)

print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4, zero_division=0
))
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, average="weighted", zero_division=0))
print("F1-score :", f1_score(y_true, y_pred, average="weighted", zero_division=0))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid()
plt.show()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()

if len(class_names) == 2:
    fpr, tpr, _ = roc_curve(y_true, y_probability[:, 1])
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid()
    plt.show()
```

This produces accuracy/loss curves, a classification report, accuracy,
precision, recall, F1-score, a confusion matrix, and an ROC curve.

#### 6. Download the ESP32 model

```python
from google.colab import files

files.download("model/export/model_int8.tflite")
files.download("model/export/labels.txt")
```

Use `model_int8.tflite` in the ESP32/TensorFlow Lite Micro firmware and keep
the label order from `labels.txt`.

### Train the model locally (optional)

The dataset is already arranged for supervised training:

```
archive/casting_data/casting_data/
├── train/def_front
├── train/ok_front
├── test/def_front
└── test/ok_front
```

For Windows native, use Python 3.10: TensorFlow 2.10.1 is the last native
Windows release. Python 3.12+ is not compatible with this Windows setup.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r training\requirements.txt
python training\train.py
```

The script uses the existing train/test split, trains MobileNetV2 at 96×96,
and writes `model\export\model_int8.tflite`, `labels.txt`, and test metrics.
The exported file is the quantized model intended for the ESP32 firmware.
If the dataset is elsewhere, pass its root with
`--data-dir C:\path\to\casting_data\casting_data`.

### 1. Clone the Repository
```bash
git clone https://github.com/SohamVashistha/Motor-Anomaly-Detection-TinyML
cd Motor-Anomaly-Detection-TinyML
```

### 2. Configure MQTT Broker
```bash
sudo apt install mosquitto mosquitto-clients -y
sudo mosquitto_passwd -c /etc/mosquitto/passwd mqttuser
sudo nano /etc/mosquitto/conf.d/security.conf
```
```
listener 1883 0.0.0.0
allow_anonymous false
password_file /etc/mosquitto/passwd
```
```bash
sudo systemctl restart mosquitto
```

### 3. Launch Node-RED Dashboard
```bash
sudo npm install -g --unsafe-perm node-red
node-red &
# Open http://localhost:1880
# Import dashboard/node-red-flow.json
```

### 4. Flash ESP32-CAM
```
1. Open firmware/esp32cam_firmware.ino in Arduino IDE
2. Update WiFi credentials and MQTT server IP
3. Connect IO0 to GND for flash mode
4. Board: AI Thinker ESP32-CAM | Port: COMX
5. Upload, remove IO0 wire, press Reset
6. Open Serial Monitor at 115200 baud
```

---

## Future Scope

| Feature | Description | Priority |
|---------|-------------|----------|
| Relay Motor Control | Connect relay module to ESP32 GPIO for remote start/stop from dashboard. On high-confidence anomaly, system auto-cuts motor power — converting from pure monitoring to closed-loop control | High |
| mTLS on MQTT | Certificate-based mutual authentication for encrypted data transmission | Medium |
| Grafana + InfluxDB | Time-series anomaly logging and historical trend dashboards | Medium |
| OTA Firmware Updates | Push model updates wirelessly without physical access | Medium |
| Multi-Camera Support | Monitor multiple motors from a single dashboard | Low |
| Edge TPU Deployment | Port to Google Coral for faster inference at higher FPS | Low |

---

## Industry Applications

| Industry | Application |
|----------|------------|
| Manufacturing | PCB defect detection on assembly lines |
| Power Sector | Motor and transformer health monitoring |
| Automotive | Weld quality inspection on chassis lines |
| Pharma | Pill packaging completeness detection |
| Mining | Conveyor belt spillage monitoring |

---

## Key Technical Decisions

**Why MobileNetV2 over a custom CNN?**
Pre-trained on ImageNet and fine-tuned for our domain. Transfer learning delivers stronger feature extraction with far less training data than building from scratch.

**Why int8 quantization?**
4x smaller model, faster inference, and negligible accuracy drop. Essential for fitting inside ESP32's 520KB RAM without floating point hardware acceleration.

**Why MQTT over HTTP?**
Persistent connections, minimal overhead, pub/sub pattern maps perfectly to sensor-to-dashboard IoT flow. Industry standard for embedded IoT communication.

**Why auto class weighting?**
Dataset had a 3:1 anomaly-to-normal imbalance. Without weighting, normal recall was only 34%. With auto-weighting it jumped to 78.4% — single toggle, massive impact.

---

## Repository Structure

```
Motor-Anomaly-Detection-TinyML/
|
|-- firmware/
|   |-- esp32cam_firmware.ino     <- ESP32 inference + MQTT publish
|
|-- dashboard/
|   |-- node-red-flow.json        <- Importable Node-RED flow
|
|-- mosquitto/
|   |-- security.conf             <- Broker security configuration
|
|-- model/
|   |-- README.md                 <- Edge Impulse project link + metrics
|
|-- README.md
```

---

## References

- [Edge Impulse Documentation](https://docs.edgeimpulse.com)
- [Casting Product Dataset - Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [Mosquitto MQTT Broker](https://mosquitto.org)
- [Node-RED](https://nodered.org)

---

<div align="center">

Built by [Soham Vashistha](https://github.com/SohamVashistha)

*The best AI is the one that runs where the problem is.*

</div>
