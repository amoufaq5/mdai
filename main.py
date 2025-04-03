from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
from model import create_multimodal_model
from evaluation import evaluate_using_mnemonics
import logging

app = FastAPI(title="Multimodal Diagnostic AI")

# Attempt to load the pre-trained model; fallback if necessary.
try:
    model = tf.keras.models.load_model('saved_model/diagnosis_model')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = create_multimodal_model()  # Fallback to a new model instance

class DiagnosisRequest(BaseModel):
    text_data: str          # Patient's textual input (e.g., description of symptoms)
    numerical_data: list    # List of numerical features (e.g., lab results)
    age: int = None         # ASMETHOD parameter: Age/appearance
    symptoms: str = None    # ASMETHOD parameter: Symptoms description
    time_persisting: int = 0  # ASMETHOD parameter: Duration of symptoms
    # Additional fields (e.g., medication, extra medicines, etc.) can be added as needed

@app.post("/diagnose")
async def diagnose(diagnosis_request: DiagnosisRequest, image: UploadFile = File(None)):
    try:
        # --- Text preprocessing ---
        # (For demonstration, using a dummy tokenizer that maps characters to token IDs.)
        text_sequence = [ord(c) % 5000 for c in diagnosis_request.text_data]
        text_sequence = text_sequence[:100] + [0]*(100 - len(text_sequence))
        text_sequence = np.array(text_sequence).reshape(1, -1)

        # --- Numerical data ---
        numerical_data = np.array(diagnosis_request.numerical_data).reshape(1, -1)

        # --- Image preprocessing ---
        if image:
            import cv2
            image_bytes = await image.read()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
        else:
            # If no image is provided, use a placeholder (black image)
            img = np.zeros((1, 224, 224, 3), dtype='float32')

        # --- Model Inference ---
        predictions = model.predict([text_sequence, img, numerical_data])
        # predictions is assumed to be a probability vector for the 3 output classes

        # --- Evaluation using clinical mnemonics ---
        patient_data = {
            "age": diagnosis_request.age,
            "symptoms": diagnosis_request.symptoms,
            "time_persisting": diagnosis_request.time_persisting
            # Add other ASMETHOD fields as needed.
        }
        decision = evaluate_using_mnemonics(patient_data, predictions[0])
        return {"diagnosis": decision, "model_confidence": predictions.tolist()}
    except Exception as e:
        logging.error(f"Error during diagnosis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
