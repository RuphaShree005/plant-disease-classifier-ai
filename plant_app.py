import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

# Load model
model = load_model("plant_disease_model.h5")

# Load classes
with open("class_names.json") as f:
    classes = json.load(f)

# 🌿 Disease info (Flexible matching)
disease_info = {
    "early_blight": {
        "about": "A fungal disease causing dark spots on older leaves.",
        "effect": "Reduces plant growth and crop yield.",
        "solution": "Remove infected leaves and apply fungicide."
    },
    "late_blight": {
        "about": "A fast-spreading disease in wet conditions.",
        "effect": "Can destroy the plant quickly.",
        "solution": "Avoid moisture and use copper-based sprays."
    },
    "septoria": {
        "about": "Septoria leaf spot causes small circular spots with dark edges.",
        "effect": "Leads to yellowing and dropping of leaves.",
        "solution": "Remove infected leaves and apply fungicide."
    },
    "healthy": {
        "about": "The plant shows no disease symptoms.",
        "effect": "Healthy growth and good yield.",
        "solution": "Maintain proper care and monitoring."
    }
}

# 🔥 Function to match class name intelligently
def get_disease_info(predicted_class):
    name = predicted_class.lower()

    for key in disease_info:
        if key in name:
            return disease_info[key]

    return None

# Page setup
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;color:green;'>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)

st.info("📌 Upload a plant leaf image to detect disease, understand its impact, and get solutions.")

# Upload
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = classes[class_index]

    # 🌟 Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Result", "📖 Details", "💡 Solution"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.markdown("## 🌿 Prediction Result")
        st.success(f"Detected: {predicted_class}")

        st.markdown("### 📊 Confidence Level")
        st.progress(int(confidence))

        # ✅ SMART MESSAGE
        if confidence > 85:
            st.success(f"✅ Very High Confidence ({confidence:.2f}%)")
        elif confidence > 60:
            st.warning(f"⚠️ Moderate Confidence ({confidence:.2f}%)")
        else:
            st.error(f"❗ Low Confidence ({confidence:.2f}%)")
            st.write("👉 The model is confused. Try uploading a clearer image or different angle.")

        # 📊 Chart explanation
        st.markdown("### 📈 Prediction Breakdown")

        probs = prediction[0]
        chart_data = {
            classes[i]: float(probs[i]) * 100 for i in range(len(classes))
        }

        st.bar_chart(chart_data)

        st.info("📌 This chart shows how the model is comparing different diseases. Higher value = higher chance.")

    # ---------------- TAB 2 ----------------
    with tab2:
        st.markdown("## 📖 Disease Explanation")

        info = get_disease_info(predicted_class)

        if info:
            st.info("🧠 What is this disease?")
            st.write(info["about"])

            st.warning("🌿 How it affects the plant?")
            st.write(info["effect"])

            st.markdown("### 🌡 Severity Analysis")

            if confidence > 85:
                st.error("🔴 High - Disease may spread quickly")
            elif confidence > 60:
                st.warning("🟡 Medium - Needs attention")
            else:
                st.success("🟢 Low - Mostly safe")
        else:
            st.warning("⚠️ Detailed info not found for this class, but prediction is still shown above.")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.markdown("## 💡 Recommended Actions")

        if info:
            st.success(info["solution"])

            st.markdown("### 🚜 Additional Tips")
            st.write("""
            ✔ Ensure proper sunlight  
            ✔ Avoid overwatering  
            ✔ Regularly inspect leaves  
            ✔ Remove infected parts early  
            """)
        else:
            st.warning("General Advice: Keep plant clean, avoid excess water, and monitor regularly.")

    # ---------------- EXTRA ----------------
    st.markdown("---")

    if confidence < 50:
        st.error("🚨 The model is unsure. Please upload a clearer leaf image for better accuracy.")
    else:
        st.success("✅ Analysis complete. You can try another image for comparison.")