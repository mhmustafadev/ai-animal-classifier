
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="AI Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

@st.cache_resource
def load_models():
    with open("animal_model.pkl", "rb") as f:
        classifier = pickle.load(f)

    # Downloads ImageNet weights automatically on first startup
    model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )

    return classifier, model

translations = {
    "gatto": "Cat",
    "cane": "Dog",
    "cavallo": "Horse",
    "pecora": "Sheep",
    "elefante": "Elephant",
    "scoiattolo": "Squirrel",
    "gallina": "Chicken",
    "ragno": "Spider",
    "mucca": "Cow",
    "farfalla": "Butterfly"
}

st.title("🐾 AI Animal Classifier")
st.write("Upload an animal image and let AI identify it.")

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Identify Animal"):
        with st.spinner("AI is analyzing the image..."):
            try:
                classifier, model = load_models()

                resized = image.resize((224, 224))
                array = np.array(resized, dtype=np.float32)

                batch = preprocess_input(
                    np.expand_dims(array, axis=0)
                )

                features = model.predict(batch, verbose=0)
                probabilities = classifier.predict_proba(features)[0]

                top_indices = np.argsort(probabilities)[::-1][:3]

                best_index = top_indices[0]
                best_class = classifier.classes_[best_index]
                best_name = translations.get(best_class, best_class)

                st.success(f"Prediction: {best_name}")

                st.subheader("Top 3 Predictions")

                for index in top_indices:
                    animal = classifier.classes_[index]
                    name = translations.get(animal, animal)
                    confidence = float(probabilities[index])

                    st.write(f"**{name}: {confidence:.2%}**")
                    st.progress(confidence)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
