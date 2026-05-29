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
