# 🚧 RoadWatch — AI-Powered Road Damage Reporting System

RoadWatch is an AI-powered civic-tech platform designed to help citizens report road damage efficiently. The system combines **Deep Learning**, **Location Intelligence**, and **LLM-assisted authority mapping** to detect potholes and road defects, identify the responsible government authority, and generate complaint-ready reports.

---

## 📌 Problem Statement

Road damage such as potholes and cracks often remain unattended due to:

* Lack of awareness about the responsible authority
* Manual complaint processes
* Poor reporting workflows
* Delays in identifying jurisdiction and escalation channels

RoadWatch aims to simplify this process using AI and automation.

---

# ✨ Features

### 🛣️ Road Damage Detection

* Deep Learning-based detection of:

  * Potholes
  * Longitudinal cracks
  * Transverse cracks
  * Alligator cracks

Uses a custom-trained YOLO model for real-time road defect analysis.

---

### 📍 Live Location Detection

* Uses **Nominatim / Geolocation APIs**
* Detects user's current location
* Extracts:

  * State
  * City
  * Coordinates

---

### 🏗️ Road-Type Classification Input

Users can specify road category:

* NH — National Highway
* SH — State Highway
* MDR — Major District Road
* City Road
* Village / Rural Road

---

### 🧠 Authority Identification Engine

RoadWatch identifies the responsible authority based on:

* User location
* Road category
* Government datasets
* Rule-based routing + RAG-assisted information retrieval

Examples:

| Road Type | Authority                  |
| --------- | -------------------------- |
| NH        | NHAI                       |
| SH        | State PWD                  |
| MDR       | District Administration    |
| City      | Municipal Corporation      |
| Rural     | NRIDA / Local Rural Bodies |

---

### 🤖 LLM-Powered Assistant

Integrated conversational assistant that:

* Answers road complaint questions
* Provides authority contact details
* Suggests escalation procedures
* Explains complaint process

Powered using:

* Groq API
* Llama models
* Retrieval-Augmented Generation (RAG)

---

### 📝 Complaint Draft Generation

The system generates complaint-ready information including:

* Damage type
* Location
* Road category
* Responsible authority
* Complaint summary

Current version creates **draft complaint records** for user review.

---

# 🏗️ System Architecture

```text
Frontend
    ↓
Location + User Input
    ↓
Deep Learning Model
    ↓
Authority Mapping Engine
    ↓
RAG + LLM Assistant
    ↓
Complaint Draft Generation
    ↓
Backend API
```

---

# ⚙️ Tech Stack

## Frontend

* HTML
* CSS
* JavaScript
* Geolocation APIs
* Nominatim

## Backend

* Python
* Flask
* Flask-CORS

## AI / ML

* YOLO (Ultralytics)
* PyTorch
* ONNX

## LLM / RAG

* Sentence Transformers
* FAISS
* Groq API
* Llama Models

## Data

* JSON datasets
* Government authority records
* Road datasets

---

# 📂 Project Structure

```text
RoadWatch/
│
├── frontend/
│
├── backend/
│   ├── main.py
│   ├── rag_engine.py
│   ├── detector.py
│   ├── data/
│   │   ├── roads.json
│   │   ├── govt_bodies.json
│
├── models/
│   ├── Best_model.pt
│   ├── Best_model.onnx
│
└── README.md
```

---

# ⚙ About the DL model 
The Model was built on kaggle notebooks using the RDD2022 and Pothole Detection Dataset (By Raj Dalsaniya).
The model was made using the YOLOv8n model by Ultralytics, where the model was trained using 4 classes 
├── class 0 : Longitudinal Crack
├── class 1 : Transverse Crack
├── class 2 : Alligator Crack
└── class 3 : Pothole
 

 here are a few snippets that were used to build the model 

``` Depnedencies install

!pip install ultralytics --quiet
!pip install opencv-python-headless --quiet
import os
import shutil
import random
import yaml
import zipfile
from pathlib import Path


import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2

import torch
import ultralytics
from ultralytics import YOLO

import glob

import urllib.request
```

``` Load dataset
DATASET_PATH = "/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT"
for root, dirs, files in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:  # don't go too deep
        subindent = ' ' * 2 * (level + 1)
        for f in files[:5]:
            print(f'{subindent}{f}')
```

``` Training the model
model = YOLO('yolov8n.yaml').load('yolov8n.pt')
results = model.train(
    data=yaml_path,
    epochs=50,           
    imgsz=640,           
    batch=16,            
    device=0,          
    project='/kaggle/working',
    name='detector_v1',
    patience=15,         
    save=True,
    plots=True,          
    val=True,
    workers=2,
    lr0=0.01,         
    lrf=0.01,           
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    augment=True,        
    degrees=5.0,         
    translate=0.1,
    scale=0.5,
    fliplr=0.5,          
    mosaic=1.0,          
)
 
print("\n✅ Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
 

```

---


# 🚀 Installation

## Clone Repository

```bash
git clone <repo-link>
cd RoadWatch
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Backend

```bash
python main.py
```

---

## Run Frontend

Example:

```bash
npm install
npm start
```

or open frontend locally depending on setup.

---

# 🧪 Model Information

RoadWatch uses a custom fine-tuned YOLO model trained on:

* Road crack datasets
* Pothole datasets
* Additional fine-tuning for pothole detection

Supported classes:

| Class ID | Damage Type        |
| -------- | ------------------ |
| 0        | Longitudinal Crack |
| 1        | Transverse Crack   |
| 2        | Alligator Crack    |
| 3        | Pothole            |

---

# 📌 Current Status

✅ Deep Learning model integrated
✅ LLM assistant integrated
✅ Location detection working
✅ Authority mapping engine integrated
✅ Complaint draft generation working
⚠️ Live government complaint submission under evaluation

---

# 🔮 Future Improvements

* Live complaint submission APIs
* Offline complaint syncing
* Mobile application
* Better jurisdiction mapping
* Multi-language support
* Severity estimation and prioritization
* Real-time road condition analytics

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Fork the repository and create a pull request.

---

# 📄 License

This project is intended for educational, research, and civic-tech innovation purposes.

---

## Built with AI, code, and a lot of debugging.
