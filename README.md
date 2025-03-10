# 🚀 Helmet Violation Detector

A **real-time helmet violation detection system** that identifies motorcycles, detects whether riders are wearing helmets, and reads license plates using **YOLO, PaddleOCR, and OpenCV**.

---

### 📦 `Python 3.12 or below` is required for compilation

## 📌 Features

- 🏍️ **Motorcycle Detection** (YOLO-based)
- 🎩 **Helmet Detection** (Custom YOLO model)
- 🚶 **Person-Motorcycle Association** (using IoU & aspect ratio similarity)
- 🔍 **License Plate Recognition** (PaddleOCR)
- 🚗 **License Plate Detection** (YOLO-based)
- 🎥 **Real-time Camera Processing** (OpenCV)
- 📊 **Report Generation in Excel** (OpenPyXL)
- 🖥️ **GUI Interface** (PyQt6)

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://forge.bitmutex.com/bigwiz/new-plate
cd helmet-violation-detector
```

### 2️⃣ Set Up Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate.bat    # On Windows CMD
.\venv\Scripts\activate.ps1    # On Powershell
```

### 3️⃣ Install Dependencies

Ensure you have **Python 3.8+** installed. Then, run:

```bash
pip install -r requirements.txt
```

**🔹 Note:**
1. If you face **PaddleOCR-related dependency issues**, make sure to install **paddlepaddle**:

```bash
pip install paddlepaddle
```

2. If you face **Protobuf related issues**, make sure to install correct version of protobuf (4.25.*):

```bash
pip install protobuf==4.25.*
```

---


## 🚀 Usage

### 🔹 GUI Mode

To launch the **Helmet Violation Detector GUI**, run:

```bash
python main.py
```

### 🔹 Real-time Camera Detection

Click **"Start Real-Time Analysis"** in the GUI to process frames from your webcam.

### 🔹 Processing Images/Videos

1. Click **"Upload Media"** and select an image or video.
2. Click **"Analyze Media"** to process.
3. Click **"Create Report"** to generate an **Excel report**.
4. Click **"Start Real-Time Mode"** to start/stop **analysis on connected webcam/camera feed**.

---

## 📝 Limitations

-  Cannot do excel writes in real time mode , however console outs detected number plates at all times
-  Model Accuracy depends on presented input source quality and fidelity.
-  Currently motorcycle to person association is done however,associations for person to helmet is also required.

---

## ⚙️ Dependencies

This project requires:

- **Python 3.8+**
- **YOLO (Ultralytics)**
- **PaddleOCR**
- **OpenCV**
- **Mediapipe**
- **PyQt6**
- **OpenPyXL (Excel support)**

You can install them using:

```bash
pip install -r requirements.txt
```

---

## 📌 Troubleshooting

### ❓ Facing PaddleOCR Issues?

If you get errors related to `protobuf` or `paddleocr`, try:

```bash
pip install paddlepaddle
```

or

```bash
pip install protobuf==3.20.*
```

---
