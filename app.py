
import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# ---------------- TRANSLATIONS ----------------

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

# ---------------- FIND MODEL FILE ----------------

BASE_DIR = Path(__file__).resolve().parent

def find_classifier_file():
    # Prefer the original filename
    exact_file = BASE_DIR / "animal_model.pkl"

    if exact_file.exists():
        return exact_file

    # Also accept names such as animal_model (1).pkl
    matching_files = list(BASE_DIR.glob("animal_model*.pkl"))

    if matching_files:
        return matching_files[0]

    raise FileNotFoundError(
        "Animal model not found. Upload animal_model.pkl "
        "to the root of your GitHub repository."
    )

# ---------------- LOAD AI MODELS ----------------

@st.cache_resource
def load_models():

    classifier_path = find_classifier_file()

    with open(classifier_path, "rb") as file:
        classifier = pickle.load(file)

    # Load pretrained ResNet50
    model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )

    return classifier, model

# ---------------- APPLICATION UI ----------------

st.title("🐾 AI Animal Classifier")

st.write(
    "Upload an animal image and let Artificial "
    "Intelligence identify it."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Your Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Identify Animal", use_container_width=True):

        with st.spinner("AI is analyzing your image..."):

            try:
                # Load classifier and ResNet50
                classifier, model = load_models()

                # Resize image
                resized_image = image.resize((224, 224))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                # Preprocess image for ResNet50
                image_batch = np.expand_dims(
                    image_array,
                    axis=0
                )

                processed_image = preprocess_input(
                    image_batch
                )

                # Extract image features
                features = model.predict(
                    processed_image,
                    verbose=0
                )

                # Predict animal
                probabilities = classifier.predict_proba(
                    features
                )[0]

                # Get top three predictions
                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]

                best_index = top_indices[0]
                best_class = classifier.classes_[best_index]

                best_animal = translations.get(
                    best_class,
                    best_class
                )

                # Display result
                st.success(
                    f"Prediction: {best_animal} 🐾"
                )

                st.subheader("Top 3 Predictions")

                for index in top_indices:

                    animal_class = classifier.classes_[index]

                    animal_name = translations.get(
                        animal_class,
                        animal_class
                    )

                    confidence = float(
                        probabilities[index]
                    )

                    st.write(
                        f"**{animal_name}: {confidence:.2%}**"
                    )

                    st.progress(confidence)

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )

st.divider()

st.caption(
    "AI Animal Classifier | Built with Python, "
    "TensorFlow, ResNet50 and Streamlit"
)
