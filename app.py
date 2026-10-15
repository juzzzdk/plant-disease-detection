import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tf_keras as keras

st.set_page_config(page_title="AI AgroCare - Multi-Crop Disease Detector", page_icon="🌿")

st.title("🌿 AI AgroCare: Multi-Crop Disease Detector")
st.write("Blossom Public School Science Expo '26")
st.markdown("---")

@st.cache_resource
def load_disease_model():
    model = keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_disease_model()

# Remedies dictionary matching your 8 classes
remedies = {
    "Tomato Healthy": {
        "status": "Healthy Tomato Leaf",
        "color": "green",
        "advice": "Plant is healthy! Ensure consistent soil moisture and avoid splashing water on leaves."
    },
    "Tomato Early Blight": {
        "status": "Diseased (Tomato Early Blight)",
        "color": "orange",
        "advice": "1. Prune affected lower leaves.\n2. Apply organic neem oil or copper-based fungicide spray every 7–10 days.\n3. Mulch around the plant base to prevent soil splash."
    },
    "Tomato Bacterial Spot": {
        "status": "Diseased (Tomato Bacterial Spot)",
        "color": "red",
        "advice": "1. Apply copper spray combined with mancozeb.\n2. Sanitize pruning shears between plants.\n3. Avoid handling foliage while wet."
    },
    "Potato Healthy": {
        "status": "Healthy Potato Leaf",
        "color": "green",
        "advice": "Plant is healthy! Maintain regular watering, proper soil drainage, and full sunlight."
    },
    "Potato Early Blight": {
        "status": "Diseased (Potato Early Blight)",
        "color": "orange",
        "advice": "1. Remove heavily spotted lower leaves.\n2. Apply copper fungicide every 7–10 days.\n3. Ensure good airflow between plants."
    },
    "Potato Late Blight": {
        "status": "Diseased (Potato Late Blight)",
        "color": "red",
        "advice": "1. Apply systemic fungicide immediately.\n2. Destroy severely infected plants to prevent spore spread.\n3. Keep foliage dry during watering."
    },
    "Bellpepper Healthy": {
        "status": "Healthy Bell Pepper Leaf",
        "color": "green",
        "advice": "Plant is healthy! Keep soil consistently moist and supply balanced organic fertilizer."
    },
    "Bellpepper Bacterial Spot": {
        "status": "Diseased (Bell Pepper Bacterial Spot)",
        "color": "red",
        "advice": "1. Spray copper hydroxide bactericide.\n2. Space plants adequately to encourage fast drying.\n3. Avoid working in fields when plants are wet."
    }
}

uploaded_file = st.file_uploader("Upload a Leaf Photo (JPG/PNG):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    
    with st.spinner("Analyzing leaf patterns..."):
        size = (224, 224)
        processed_image = ImageOps.fit(image.convert("RGB"), size, Image.Resampling.LANCZOS)
        image_array = np.asarray(processed_image)
        normalized = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized

        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        raw_label = class_names[index]
        clean_label = " ".join(raw_label.split()[1:]) if raw_label[0].isdigit() else raw_label
        confidence = float(prediction[0][index]) * 100

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Condition", clean_label)
    with col2:
        st.metric("Confidence Score", f"{confidence:.1f}%")

    if clean_label in remedies:
        info = remedies[clean_label]
        if info["color"] == "green":
            st.success(f"**Status:** {info['status']}\n\n**Advice:**\n{info['advice']}")
        elif info["color"] == "orange":
            st.warning(f"**Status:** {info['status']}\n\n**Recommended Remedy:**\n{info['advice']}")
        else:
            st.error(f"**Status:** {info['status']}\n\n**Recommended Remedy:**\n{info['advice']}")
    else:
        st.info("Analysis complete. Check that labels match class names.")
