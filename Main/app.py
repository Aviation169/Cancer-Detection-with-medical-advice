import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fpdf import FPDF
from PIL import Image
import io

# --- CNN Model Parameters ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 4
CLASS_NAMES = ['begin', 'early', 'pre', 'post']

# --- Load LLaMA-3B Model ---
model_name = "g:\My Drive\llama-3.2-3b-instruct"  # Update this path
tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
llama_model.to(device)

# 1. Load the trained CNN model
@st.cache_resource
def load_trained_model(model_path='cancer_stage_classifier_vgg16_final.h5'):
    try:
        model = load_model(model_path)
        st.success("CNN Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

# 2. Predict on an image
def predict_image(model, img):
    try:
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None, None

# 3. Get medical advice from LLaMA
def get_medical_advice(stage):
    prompt = f"""
    You are a medical AI assistant. The patient has been diagnosed with {stage} stage cancer.
    
    Provide the following details:
    1️⃣ *Stage Meaning* – Explain what this stage means.
    2️⃣ *Treatment Plan* – List recommended treatments.
    3️⃣ *Diet Plan* – Suggest foods to eat/avoid.
    4️⃣ *Precautions* – Provide health precautions.

    Respond in a professional but simple way.

    ### Response:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llama_model.generate(**inputs, max_length=1024)
    
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    split_output = full_output.split("### Response:")
    ai_response = split_output[1].strip() if len(split_output) > 1 else full_output.strip()
    
    return ai_response

# 4. Generate PDF Report with Unicode support
def generate_pdf_report(stage, ai_response):
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, "Medical Report", ln=True, align="C")
            self.ln(10)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add cancer stage
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Cancer Stage: {stage.upper()}", ln=True)
    pdf.ln(5)

    # Try to add a Unicode font, fallback to Arial if not available
    try:
        # Update this path for Windows if needed, e.g., 'C:/Windows/Fonts/arial.ttf'
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font("DejaVu", size=11)
    except:
        pdf.set_font("Arial", size=11)
        # Replace problematic characters if no Unicode font is available
        ai_response = ai_response.encode('ascii', 'ignore').decode('ascii')

    pdf.multi_cell(0, 10, ai_response)

    # Output to BytesIO
    pdf_buffer = io.BytesIO()
    pdf_output = pdf.output(dest='S')  # Get PDF as string (already Unicode-aware with uni=True font)
    pdf_buffer.write(pdf_output.encode('utf-8'))  # Encode as UTF-8
    pdf_buffer.seek(0)
    
    pdf_filename = f"cancer_{stage}_report.pdf"
    return pdf_buffer, pdf_filename

# --- Streamlit UI ---
def main():
    st.title("Cancer Stage Classifier and Medical Advisor")
    st.write("Upload an image to classify the cancer stage and get medical advice.")

    # Load the model
    cnn_model = load_trained_model()
    if cnn_model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Predict stage
        with st.spinner("Classifying..."):
            predicted_class, confidence = predict_image(cnn_model, img)
        
        if predicted_class:
            st.subheader("Prediction Results")
            st.write(f"**Predicted Stage:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.4f}")

            # Get medical advice
            with st.spinner("Generating medical advice..."):
                ai_response = get_medical_advice(predicted_class)
            
            st.subheader("Medical Advice")
            st.text_area("AI Response", ai_response, height=300)

            # Generate and provide PDF download
            with st.spinner("Generating PDF report..."):
                pdf_buffer, pdf_filename = generate_pdf_report(predicted_class, ai_response)
            
            st.download_button(
                label="Download Medical Report",
                data=pdf_buffer,
                file_name=pdf_filename,
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()