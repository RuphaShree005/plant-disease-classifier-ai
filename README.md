# plant-disease-classifier-ai
AI-powered plant disease detection system using CNN with an interactive Streamlit dashboard, confidence analysis, and user-friendly disease explanations.

# 🌿 Explainable Plant Disease AI

An AI-powered web application that detects plant diseases from leaf images using Deep Learning (CNN) and provides user-friendly explanations, confidence analysis, and treatment suggestions.

---

# Features

-  Plant disease detection using CNN
-  Confidence score with visual progress bar
-  Prediction breakdown chart (probability for each class)
-  Explainable AI (what the disease is & how it affects plants)
-  Severity level analysis (Low / Medium / High)
-  Smart treatment and prevention suggestions
-  Interactive and user-friendly Streamlit UI

---

# Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- PIL (Image Processing)

---

# Project Structure
plant-disease-project/
│
├── dataset
├── training.py
├── plant_app.py 
├── plant_disease_model.h5 
├── class_names.json 

# How It Works

1. Upload a leaf image 🌿  
2. Model predicts disease using CNN  
3. Displays:
   - Disease name  
   - Confidence level  
   - Severity analysis  
   - Probability chart  
4. Provides:
   - Disease explanation  
   - Impact on plant  
   - Suggested solutions  

---

# Example Output

-  Detected Disease: Tomato Septoria Leaf Spot  
-  Confidence: 92%  
-  Severity: High  
-  Solution: Apply fungicide and remove infected leaves  

---

# Future Improvements

-  Multi-language support (Tamil, Hindi)
-  Voice-based explanation
-  Prediction history dashboard
-  Mobile-friendly UI
-  Live camera detection

- Helps farmers identify plant diseases early  
- Supports smart agriculture  
- Reduces crop loss with timely action  



