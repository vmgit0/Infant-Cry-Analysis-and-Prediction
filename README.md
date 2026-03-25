## ⚠️ Notes

- Recommended Python version: **3.10**
- PyAudio may not work on Python 3.12+

---

# 🎧 Infant Cry Analysis and Prediction System - ACADEMIC PROJECT

An AI-powered Final Year Engineering Project to analyze infant cry audio and predict the reason (hunger, discomfort, pain, tiredness) using Machine Learning.

---

## 📌 Problem Statement

Infants communicate through crying, but identifying the reason is difficult.  
This project uses AI to classify cry sounds and assist caregivers.

---

## 🚀 Features

- 🎤 Audio input (microphone / dataset)
- 🔊 Noise reduction
- 📊 MFCC feature extraction
- 🤖 Random Forest classification
- 📈 Audio visualization
- 🌐 Flask-based web application (localhost)
- 🧠 Real-time prediction with probability distribution of cry reasons

---

## 🌐 Web Application (Flask)

A web-based interface is developed using **Flask** that runs on **localhost**.

### Features:
- 🎙️ Record audio directly from microphone
- 📂 Upload audio file option
- ⚡ Real-time prediction
- 📊 Displays probability distribution of different cry types (hungry, pain, tired, discomfort)

### Run the app:

python backend/app.py

Then open in browser:

http://127.0.0.1:5000/

---

## 🧠 Machine Learning Approach

- Feature Extraction: MFCC (Mel Frequency Cepstral Coefficients)
- Model: Random Forest Classifier
- Output: Predicted class + probability distribution
- Evaluation: Accuracy, Confusion Matrix

---

## 🏗️ Project Structure

Infant-Cry-Analysis-and-Prediction/
├── audio_dataset/
├── backend/
│   ├── app.py
│   └── Cry_Model.ipynb
├── templates/
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

git clone https://github.com/YOUR_USERNAME/Infant-Cry-Analysis-and-Prediction.git  
cd Infant-Cry-Analysis-and-Prediction  

python -m venv venv  
venv\Scripts\activate  

pip install -r requirements.txt  

python backend/app.py  

---

## 📊 Tech Stack

Python, Flask, NumPy, Pandas, Librosa, Scikit-learn, Matplotlib, Seaborn

---


## 🎯 Conclusion

This project demonstrates how AI + audio processing can help understand infant needs and improve caregiving.

