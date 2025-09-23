
# 📄 Document Classifier (Aadhaar, PAN, Other)

This project is a **deep learning–based document classification system** that uses a **Convolutional Neural Network (CNN)** to automatically classify scanned documents into categories:
- 🪪 **AADHAAR**
- 🧾 **PAN**
- 📂 **OTHER**

Built with **PyTorch** and deployed using **Streamlit**.

---

## ✨ Features
- ✅ Train a CNN (ResNet18 backbone) on custom document datasets.  
- ✅ Data augmentation for robust training.  
- ✅ Early stopping and learning rate scheduling.  
- ✅ Inference with confidence scores and prediction time.  
- ✅ Streamlit UI for easy document upload and testing.  

---

## 📂 Project Structure
```
├── data/                 # Dataset folder (ignored in git)
│   ├── train/            # Training images
│   └── val/              # Validation images
├── models/               # Trained model weights (.pth)
|__ logs             
├── src/
│   ├── logger.py         # Logging utility
│   └── cnn_classifier.py # CNN inference class
├── app.py                # Streamlit app
├── train_cnn.py          # CNN training script            
├── requirements.txt      # Dependencies
├── .gitignore            # Ignore datasets & models
└── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone repo
```bash
git clone https://github.com/rahulsm067/Doccument_classifier-Aadhaar-vs-PAN
cd  Doccument_classifier-Aadhaar-vs-PAN
```

### 2. Create virtual environment
```bash
conda create -n docenv
conda activate docenv    
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training

Make sure your dataset is organized like this:
```
data/train/
    Aadhaar/
    PAN/
    Other/
data/val/
    Aadhaar/
    PAN/
    Other/
```

Run training:
```bash
python src/train_cnn.py
```

The best model will be saved at:
```
models/cnn_model.pth
```

---


## 🌐 Deployment (Streamlit)

Run the app locally:
```bash
streamlit run app.py
```

