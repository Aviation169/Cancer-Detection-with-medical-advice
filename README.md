Cancer Stage Classification and Medical Advice System

(! A GPU is required to run this code as it utilizes the LLama model for medical advice.)
This repository contains two Python scripts designed to classify cancer stages from images using a Convolutional Neural Network (CNN) with transfer learning (VGG16) and provide automated medical advice using a language model (LLaMA-3B). The system includes model training, prediction, and report generation capabilities.

Script 1: Trains a CNN model to classify cancer stages (begin, early, pre, post) using images from specified directories then loads the trained model, predicts cancer stages for single or batch images, generates medical advice, and creates PDF reports without UI
Script 2: Loads the trained model, predicts cancer stages for single or batch images, generates medical advice, and creates PDF reports with Streamlit UI.

1> Features->
Image Classification: Uses a pre-trained VGG16 model fine-tuned to classify cancer stages from images.
Data Augmentation: Applies random flips, rotations, and brightness adjustments to improve model robustness.
Medical Advice: Integrates LLaMA-3B to generate detailed medical advice based on predicted stages.
Report Generation: Creates PDF reports with predictions and advice.
Batch Processing: Supports single-image and batch predictions.

2> Requirements->
Python 3.8+
TensorFlow 2.x
NumPy
Matplotlib
Transformers (Hugging Face)
PyTorch (for LLaMA)
FPDF (for PDF generation)
A compatible GPU (optional but recommended for faster training/prediction)
Install dependencies using: (pip install tensorflow numpy matplotlib transformers torch fpdf)

3> Setup->
Clone the Repository: 
git clone <repository-url>
cd <repository-folder>

Prepare Data:
Organize your training images into four folders: begin, early, pre, and post.
Supported formats: .png, .jpg, .jpeg.

Install Fonts (Optional):
For PDF generation with special characters, ensure the FreeSans font is available (e.g., /usr/share/fonts/truetype/freefont/FreeSans.ttf on Linux). Adjust the path in Script 2 if needed.

Download LLaMA-3B:
Update the model_name path in Script 2 to point to your local LLaMA-3B model (e.g., "g:/My Drive/llama-3.2-3b-instruct").

4> Directory Structure->

Here you can download the CNN model file: https://drive.google.com/file/d/1rzeaO0Yjkw_FjPsWsnTrzrI7opbU5Mi9/view?usp=sharing

Here you can download LLama3.2 3B model-> (option 1: Directly load the model from Hugging face, and option 2: request me on drive to download the LLama model).
<i> Hugging face: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct -> 

[from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")]

<ii> Drive: https://drive.google.com/drive/folders/1EUpuE5uDAIozW-Sqh6mquE7RJibQDvDf?usp=sharing

Here you can download the dataset: https://drive.google.com/drive/folders/1dYEVn-zYiWhwXm2HdXKiVadFcSjFD3Iw?usp=sharing

5> Output
Training:
Saved models: best_model.h5 (best weights) and cancer_stage_classifier_vgg16_final.h5 (final model).
Console logs with training/validation metrics.
Prediction:
Console output: Predicted class, confidence, and medical advice.
Visual output: Matplotlib plot of the image with prediction.
PDF report: Detailed report with stage and advice.

