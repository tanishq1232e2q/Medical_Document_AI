# 💊 MedAI: LLM Enhanced Hybrid Framework for Clinical Document Error Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![OCR](https://img.shields.io/badge/OCR-EasyOCR%20%7C%20TrOCR-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue)

**An AI-powered clinical decision support system that detects medication errors, prescription contradictions, dosage issues, and drug interactions using OCR, NLP, Rule-Based Reasoning, and Large Language Models.**

</div>

---

## 📌 Overview

Medication errors remain one of the leading causes of preventable healthcare incidents. Handwritten prescriptions, incorrect dosages, drug interactions, and contraindications often go unnoticed until they reach the patient.

**MedGuard AI** is an intelligent prescription analysis system that automatically extracts information from prescriptions and detects potential medication safety issues using a hybrid AI architecture consisting of:

* OCR for prescription digitization
* Biomedical Named Entity Recognition (NER)
* Rule-based clinical reasoning
* Large Language Models (LLMs)
* Drug knowledge base validation

The system assists healthcare professionals by identifying potentially dangerous prescriptions before medication is dispensed.

---

# 🚀 Features

### 📄 Prescription OCR

* EasyOCR for handwritten prescriptions
* TrOCR fallback for difficult handwriting
* Multiple preprocessing strategies
* Image enhancement and deskewing

### 🧠 Biomedical NLP

* BioBERT-based Named Entity Recognition
* Drug extraction
* Disease extraction
* Dosage extraction
* Medical test identification
* Symptom recognition

### 💊 Clinical Rule Engine

Detects:

* ✅ Drug overdose
* ✅ Drug underdose
* ✅ Drug interactions
* ✅ Contraindications
* ✅ Frequency conflicts
* ✅ Duplicate therapies
* ✅ Brand-to-generic conversion

---

### 🤖 LLM Clinical Reasoning

Uses a fine-tuned **FLAN-T5** model to identify medication errors beyond predefined rules, including:

* Hidden contradictions
* Context-aware reasoning
* Multi-drug safety analysis
* Prescription interpretation

---

### 📚 Drug Knowledge Base

Includes:

* Safe dosage ranges
* Drug interaction database
* Contraindication rules
* Brand-to-generic mappings
* Medication normalization

---

# 🏗️ System Architecture

```
Prescription Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 EasyOCR / TrOCR
        │
        ▼
 Extracted Text
        │
 ┌───────────────┐
 │               │
 ▼               ▼
BioBERT NER   Rule Engine
 │               │
 ▼               ▼
Medical       Drug Safety
Entities      Validation
 │               │
 └──────┬────────┘
        ▼
 Fine-Tuned FLAN-T5
        │
        ▼
 Safety Report
```

---

# ⚙️ Technologies Used

| Category         | Technology                    |
| ---------------- | ----------------------------- |
| Programming      | Python                        |
| Deep Learning    | PyTorch                       |
| NLP              | HuggingFace Transformers      |
| OCR              | EasyOCR, TrOCR, Tesseract OCR |
| Biomedical NLP   | BioBERT                       |
| LLM              | FLAN-T5                       |
| Fine-Tuning      | PEFT / LoRA                   |
| Image Processing | OpenCV, Pillow                |
| Data Analysis    | NumPy, Scikit-learn           |
| Visualization    | Matplotlib                    |

---

# 📂 Project Structure

```
MedGuard-AI/
│
├── models/
│   ├── BioBERT
│   ├── FLAN-T5
│   └── TrOCR
│
├── dataset/
│
├── knowledge_base/
│   ├── drug_database.json
│   ├── dosage_rules.json
│   └── interactions.json
│
├── images/
│
├── notebooks/
│
├── outputs/
│
├── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔍 Medication Safety Checks

The system automatically detects:

### Overdose

```
Amlodipine 50 mg

❌ Maximum recommended dose: 10 mg
```

---

### Underdose

```
Paracetamol 100 mg

❌ Below therapeutic dose
```

---

### Drug Interactions

Example:

```
Warfarin + Ibuprofen

⚠ Increased bleeding risk
```

---

### Contraindications

Example:

```
Metformin

Patient:
eGFR <30

⚠ Contraindicated
```

---

### Frequency Conflicts

Example:

```
Take once daily

Take every 6 hours

⚠ Conflicting dosage instructions
```

---

# 📊 Output Example

```
Prescription Analysis

Detected Drugs
--------------
Metformin
Warfarin
Ibuprofen

Detected Issues
---------------
✔ Drug Interaction
✔ Contraindication
✔ Overdose

Overall Severity
----------------
HIGH

Suggested Correction
--------------------
Discontinue Ibuprofen
Monitor INR
Reduce dosage
```

---

# 🧪 Model Components

### OCR

* EasyOCR
* TrOCR
* Tesseract (fallback)

### NLP

* BioBERT
* Biomedical NER

### Clinical Reasoning

* Fine-Tuned FLAN-T5
* Few-shot prompting
* Context-aware contradiction detection

---

# 📈 Evaluation Metrics

The system can be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* AUC
* Confusion Matrix

---

# 💡 Applications

* Hospital Pharmacy
* Electronic Health Records
* Clinical Decision Support Systems
* Telemedicine
* Medication Safety
* Prescription Validation
* Healthcare AI Research

---

# 🛠️ Installation

```bash
git clone https://github.com/yourusername/MedGuard-AI.git

cd MedGuard-AI

pip install -r requirements.txt
```

---

# ▶️ Usage

```bash
python main.py
```

The system will:

1. Read prescription images
2. Perform OCR
3. Extract medical entities
4. Detect medication errors
5. Generate a safety report

---

# Future Improvements

* REST API using FastAPI
* Web dashboard
* Mobile application
* Explainable AI module
* Multilingual prescription support
* Real-time hospital integration
* Electronic Medical Record (EMR) integration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Tanishq Palkhe**

AI • Machine Learning • Deep Learning • Computer Vision • Natural Language Processing • Healthcare AI

---

## ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐** on GitHub.
