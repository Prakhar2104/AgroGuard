# app.py
import streamlit as st
from PIL import Image
import numpy as np
import json
import joblib
import pickle
from fpdf import FPDF
import pyttsx3
import tempfile
import os
import difflib
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# At the top of app.py


# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="AgroGuard", layout="wide", initial_sidebar_state="auto")
# ---------------------------
# Custom UI Styling
# ---------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(to right, #f5fff5, #e6ffe6);
    }

    /* Title styling */
    h1 {
        color: #2e7d32 !important;
        text-align: center;
        font-size: 2.2em !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    h2, h3 {
        color: #388e3c !important;
        font-weight: 600 !important;
        border-left: 5px solid #4caf50;
        padding-left: 8px;
        margin-top: 20px;
    }

    /* Uploaded image */
    .uploadedFile {
        border: 2px solid #4caf50 !important;
        border-radius: 12px !important;
        padding: 5px;
        background-color: #ffffffcc;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #43a047 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        font-size: 1em !important;
    }
    button[kind="primary"]:hover {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
    }

    /* Info / Success / Warning messages */
    .stSuccess {
        background-color: #d9fdd3 !important;
        border-left: 5px solid #4caf50 !important;
    }
    .stWarning {
        background-color: #fff4e5 !important;
        border-left: 5px solid #ff9800 !important;
    }
    .stInfo {
        background-color: #e8f4fd !important;
        border-left: 5px solid #2196f3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------------------
# Extra UI Styling (Sidebar + Background Image)
# ---------------------------
st.markdown(
    """
    <style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fff0, #e8f5e9);
        border-right: 2px solid #4caf50;
    }
    section[data-testid="stSidebar"] h2 {
        color: #2e7d32 !important;
        font-weight: 700 !important;
        text-align: center;
    }
    section[data-testid="stSidebar"] .stNumberInput {
        background-color: #ffffffcc !important;
        border-radius: 8px !important;
        padding: 5px;
        margin-bottom: 8px;
    }

    /* Background image for main app */
    .stApp {
        background: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Add translucent card effect for content */
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# Utility / safe loaders
# ---------------------------
def safe_load_keras(path):
    try:
        model = load_model(path, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

def safe_load_joblib_or_pickle(path):
    # Try joblib then pickle
    try:
        return joblib.load(path), None
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f), None
        except Exception as e:
            return None, str(e)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

# ---------------------------
# Load models and files (paths based on your structure)
# ---------------------------
BASE = "."  # project root where app.py lives

# Plant model
MODEL_PATH = os.path.join(BASE, "models", "mobilenetv2_focal_best_epoch8.keras")
disease_model, err = safe_load_keras(MODEL_PATH)
if err:
    st.error(f"Could not load plant disease model from {MODEL_PATH}: {err}")

# class labels
CLASS_LABELS_PATH = os.path.join(BASE, "models", "class_labels.json")
class_labels, err = load_json(CLASS_LABELS_PATH)
if err:
    st.error(f"Could not load class_labels.json from {CLASS_LABELS_PATH}: {err}")
    class_labels = []
# derive plant/crop names from class labels (first token before underscore)
plant_names = ["APPLE", "RICE", "WHEAT", "POTATO", "SUGARCANE", "COTTON", "MAIZE", "GRAPES", "TOMATO"]




# disease -> treatment mapping
DISEASE_MAP_PATH = os.path.join(BASE, "notebook", "disease_mapping.json")
raw_disease_map, err = load_json(DISEASE_MAP_PATH)
if err:
    st.error(f"Could not load disease_mapping.json from {DISEASE_MAP_PATH}: {err}")
    raw_disease_map = {}

# adapt mapping: if wrapped by 'diseases' unwrap it
if isinstance(raw_disease_map, dict) and "diseases" in raw_disease_map and isinstance(raw_disease_map["diseases"], dict):
    disease_map = raw_disease_map["diseases"]
else:
    disease_map = raw_disease_map if isinstance(raw_disease_map, dict) else {}

# soil models
SOIL_MODEL_PATH = os.path.join(BASE, "soil_dataset", "models", "soil_rf_model.pkl")
SOIL_SCALER_PATH = os.path.join(BASE, "soil_dataset", "models", "soil_scaler.pkl")

soil_model, soil_err = safe_load_joblib_or_pickle(SOIL_MODEL_PATH)
soil_scaler, scaler_err = safe_load_joblib_or_pickle(SOIL_SCALER_PATH)
if soil_err:
    st.warning(f"Warning loading soil model: {soil_err}")
if scaler_err:
    st.warning(f"Warning loading soil scaler: {scaler_err}")

# voice engine (optional: may fail in some environments)
try:
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

# ---------------------------
# Helper functions
# ---------------------------
def get_treatment_list_for(disease_name, disease_map):
    """
    Robust lookup:
    - direct match
    - replace underscores <-> spaces
    - case-normalized match
    - fuzzy match using difflib (cutoff 0.6)
    Handles nested structure where value may be {'treatment': [...]} or a list directly.
    """
    if not disease_name:
        return []

    # direct
    if disease_name in disease_map:
        entry = disease_map[disease_name]
        if isinstance(entry, dict) and "treatment" in entry:
            return entry.get("treatment", [])
        if isinstance(entry, list):
            return entry
        # if other shape, try find list inside
        for v in (entry if isinstance(entry, dict) else []):
            pass

    # underscore <-> space
    alt1 = disease_name.replace("_", " ")
    alt2 = disease_name.replace(" ", "_")
    for k in (alt1, alt2):
        if k in disease_map:
            entry = disease_map[k]
            if isinstance(entry, dict) and "treatment" in entry:
                return entry.get("treatment", [])
            if isinstance(entry, list):
                return entry

    # case-insensitive match
    lower_map = {k.lower(): k for k in disease_map.keys()}
    lk = disease_name.lower()
    if lk in lower_map:
        realk = lower_map[lk]
        entry = disease_map[realk]
        if isinstance(entry, dict) and "treatment" in entry:
            return entry.get("treatment", [])
        if isinstance(entry, list):
            return entry

    # fuzzy match
    keys = list(disease_map.keys())
    close = difflib.get_close_matches(disease_name, keys, n=1, cutoff=0.6)
    if close:
        entry = disease_map[close[0]]
        if isinstance(entry, dict) and "treatment" in entry:
            return entry.get("treatment", [])
        if isinstance(entry, list):
            return entry

    # nothing
    return []

def predict_disease_from_image(pil_img):
    if disease_model is None or not class_labels:
        return "Unknown", None
    try:
        img = pil_img.convert("RGB").resize((224, 224))
        arr = keras_image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0) / 255.0
        preds = disease_model.predict(arr)
        idx = int(np.argmax(preds, axis=1)[0])
        name = class_labels[idx] if idx < len(class_labels) else "Unknown"
        confidence = float(np.max(preds))
        return name, confidence
    except Exception as e:
        return "Unknown", None

def predict_soil_quality(features_list):
    if soil_model is None or soil_scaler is None:
        return "Model not available"
    try:
        Xs = soil_scaler.transform([features_list])
        pred = soil_model.predict(Xs)[0]
        mapping = {0: "Low", 1: "Medium", 2: "High"}
        return mapping.get(int(pred), "Unknown")
    except Exception as e:
        return "Error"

def clean_text(text):
    """Remove characters not supported by latin-1 encoding (fix for FPDF)."""
    if text is None:
        return ""
    return str(text).encode("latin-1", "ignore").decode("latin-1")

def generate_pdf_with_image(disease_name, treatments, soil_quality, pil_img):
    """
    Safe PDF generator: cleans text to latin-1 and embeds a temporary PNG of the image.
    `treatments` should be an iterable of strings (if dict, we will try to extract 'treatment').
    """
    tmp = None
    try:
        # Normalize treatments to list of strings
        t_list = []
        if isinstance(treatments, dict):
            # possible structure: {'treatment': [...]} or similar
            if "treatment" in treatments and isinstance(treatments["treatment"], (list, tuple)):
                t_list = [str(x) for x in treatments["treatment"]]
            else:
                # flatten other dict forms
                for v in treatments.values():
                    if isinstance(v, (list, tuple)):
                        t_list.extend([str(x) for x in v])
        elif isinstance(treatments, (list, tuple)):
            t_list = [str(x) for x in treatments]
        elif treatments:
            t_list = [str(treatments)]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, clean_text("AgroGuard Report"), ln=True, align='C')
        pdf.ln(6)

        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, clean_text(f"Disease: {disease_name}"), ln=True)
        pdf.ln(2)
        pdf.cell(0, 8, clean_text("Treatment Suggestions:"), ln=True)
        pdf.ln(1)
        if t_list:
            for i, t in enumerate(t_list, 1):
                # use multi_cell to wrap long lines
                pdf.multi_cell(0, 7, clean_text(f"{i}. {t}"))
                pdf.ln(1)
        else:
            pdf.cell(0, 8, clean_text("No suggestions available"), ln=True)
        pdf.ln(4)

        if soil_quality:
            pdf.cell(0, 8, clean_text(f"Soil Fertility: {soil_quality}"), ln=True)
        pdf.ln(6)

        # embed image if given (pil image)
        if pil_img is not None:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                pil_img_rgb = pil_img.convert("RGB")
                pil_img_rgb.save(tmp.name, format="PNG")
                tmp.close()
                # place image with width 90mm
                pdf.image(tmp.name, x=10, y=pdf.get_y(), w=90)
            except Exception as e:
                # don't fail PDF if image embedding fails
                print("Image embedding error:", e)

        out_path = os.path.join(".", "AgroGuard_Report.pdf")
        pdf.output(out_path)
        return out_path
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

def tts(text):
    if tts_engine is None:
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception:
        pass

# ---------------------------
# UI
# ---------------------------
st.title("🌿 AgroGuard — Plant Disease & Soil Quality Checker")
st.write("Upload a leaf image to detect disease, view treatment suggestions, optionally check soil fertility, and download a PDF report.")

col1, col2 = st.columns([1, 1])
with col1:
    st.header("1) Plant Disease Detection")

    # 1) mandatory plant dropdown (user must pick before uploading)
    plant_name = st.selectbox("Select Plant Name (required)", [""] + plant_names)

    if not plant_name:
        st.info("Please select the plant/crop from the dropdown above to proceed.")
        img = None
    else:
        # 2) show uploader only after plant selected
        uploaded = st.file_uploader(f"Upload leaf image for {plant_name} (jpg/png)", type=["jpg", "jpeg", "png"])

        if uploaded:
            try:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_container_width=True)

                with st.spinner("Predicting..."):
                    disease_name, confidence = predict_disease_from_image(img)

                # optional: compare predicted crop (from label) with user-selected crop and warn
                # pred_crop = disease_name.split("_")[0] if (disease_name and disease_name != "Unknown") else None
                # if pred_crop and pred_crop.lower() != plant_name.lower():
                #     st.warning(
                #         f"Model predicted crop **{pred_crop}**, which differs from selected **{plant_name}**. "
                #         "This may indicate a mismatch or low confidence."
                #     )

                # display results
                if plant_name:
                    st.success(f"Plant: **{plant_name}**")
                if confidence is not None:
                    st.success(f"Disease: **{disease_name}**  —  Confidence: {confidence:.2%}")
                else:
                    st.success(f"Disease: **{disease_name}**")

                # get treatments robustly
                treatments = get_treatment_list_for(disease_name, disease_map)
                if treatments:
                    st.info("Treatment suggestions:")
                    for i, t in enumerate(treatments, 1):
                        st.write(f"{i}. {t}")
                else:
                    st.warning("No treatment suggestions found for this predicted class.")
                    if st.checkbox("Show available mapping keys (for debugging)"):
                        st.write(list(disease_map.keys())[:200])

                # TTS button
                if st.button("🔊 Speak Prediction"):
                    summary = f"Predicted disease {disease_name}."
                    if treatments:
                        summary += " Suggested treatments: " + " ".join(treatments[:3])
                    tts(summary)

            except Exception as e:
                st.error(f"Failed to process image: {e}")
                img = None
        else:
            img = None

with col2:
    st.header("2) Optional Soil Quality Checker")
    with st.form("soil"):
        st.write("Enter soil test values (valid ranges shown):")

        N = st.number_input("Nitrogen (N) [0–200 kg/ha]", min_value=0.0, max_value=200.0, value=50.0, format="%.2f")
        P = st.number_input("Phosphorus (P) [0–150 kg/ha]", min_value=0.0, max_value=150.0, value=20.0, format="%.2f")
        K = st.number_input("Potassium (K) [0–300 kg/ha]", min_value=0.0, max_value=300.0, value=40.0, format="%.2f")
        pH = st.number_input("pH [3–10]", min_value=3.0, max_value=10.0, value=7.0, format="%.2f")
        EC = st.number_input("Electrical Conductivity (EC) [0–5 dS/m]", min_value=0.0, max_value=5.0, value=1.0,
                             format="%.2f")
        OC = st.number_input("Organic Carbon (OC) [0–5%]", min_value=0.0, max_value=5.0, value=1.0, format="%.2f")
        S = st.number_input("Sulfur (S) [0–100 ppm]", min_value=0.0, max_value=100.0, value=10.0, format="%.2f")
        Zn = st.number_input("Zinc (Zn) [0–10 ppm]", min_value=0.0, max_value=10.0, value=1.0, format="%.2f")
        Fe = st.number_input("Iron (Fe) [0–50 ppm]", min_value=0.0, max_value=50.0, value=5.0, format="%.2f")
        Cu = st.number_input("Copper (Cu) [0–10 ppm]", min_value=0.0, max_value=10.0, value=0.5, format="%.2f")
        Mn = st.number_input("Manganese (Mn) [0–50 ppm]", min_value=0.0, max_value=50.0, value=2.0, format="%.2f")
        B = st.number_input("Boron (B) [0–5 ppm]", min_value=0.0, max_value=5.0, value=0.5, format="%.2f")

        submitted = st.form_submit_button("Predict Soil Quality")

    # After form: show results + speak button (if predicted)
    soil_quality = None
    if submitted:
        features = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]
        with st.spinner("Predicting soil fertility..."):
            soil_quality = predict_soil_quality(features)
        if soil_quality == "Model not available":
            st.error("Soil model or scaler not available on this server.")
        elif soil_quality == "Error":
            st.error("Soil prediction error.")
        else:
            st.success(f"Soil Fertility: **{soil_quality}**")

    # Speak button must be outside the form
    if submitted and soil_quality not in (None, "Model not available", "Error"):
        if st.button("🔊 Speak Soil Quality"):
            tts(f"Soil fertility is {soil_quality}")

st.markdown("---")
st.header("3) Generate PDF report")
if st.button("Generate and Download PDF report"):
    # use last predicted disease_name and treatments (if any)
    # fallback safe values
    dn = locals().get("disease_name", None)
    # better to use variables from above if uploaded
    try:
        # prefer the one from image processing block
        current_disease = locals().get("disease_name", None)
    except Exception:
        current_disease = None

    # Use the last known variables (we kept them in locals' scope above)
    # For reliability, recompute if uploaded
    if uploaded:
        disease_name_now, _ = predict_disease_from_image(Image.open(uploaded))
        treatments_now = get_treatment_list_for(disease_name_now, disease_map)
        img_for_pdf = Image.open(uploaded)
    else:
        disease_name_now = "No image provided"
        treatments_now = []
        img_for_pdf = None

    # soil_quality from form: try to retrieve; else None
    # (we created soil_quality inside the form's scope; try to read from session)
    try:
        # if the user submitted the soil form, soil_quality is defined in session state
        s_quality = soil_quality if 'soil_quality' in locals() else None
    except Exception:
        s_quality = None

    pdf_path = generate_pdf_with_image(disease_name_now, treatments_now, s_quality, img_for_pdf)
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            btn = st.download_button("📄 Download AgroGuard Report", f, file_name="AgroGuard_Report.pdf", mime="application/pdf")
        st.success("PDF ready.")
    else:
        st.error("Failed to generate PDF.")

st.write("If something doesn't show, check the console and ensure model files and JSONs are in the correct folders.")
