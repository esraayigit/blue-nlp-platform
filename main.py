import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import uvicorn
import numpy as np
import tensorflow as tf

from datetime import datetime

from transformers import AutoTokenizer

import firebase_admin
from firebase_admin import credentials, firestore

from fastapi import FastAPI
from pydantic import BaseModel

# =========================================================
# FIREBASE
# =========================================================

firebase_json = json.loads(
    os.environ["FIREBASE_CREDENTIALS"]
)

cred = credentials.Certificate(firebase_json)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# =========================================================
# TOKENIZER
# =========================================================

model_name = "ytu-ce-cosmos/turkish-small-bert-uncased"

print("Tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Tokenizer hazır.")

# =========================================================
# TFLITE MODEL
# =========================================================

print("TFLite model yükleniyor...")

interpreter = tf.lite.Interpreter(
    model_path="house_md_small.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model hazır.")

# =========================================================
# LABELS
# =========================================================

stages = [
    'değerlendirme',
    'hipotez',
    'kesin tanı',
    'tedavi',
    'test'
]

emotions = [
    'alaycı',
    'ciddi',
    'endişe',
    'korku',
    'nötr',
    'odaklanmış'
]

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()

class PatientData(BaseModel):
    text: str
    userId: str

# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "Backend çalışıyor"
    }

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():
    return {
        "ok": True
    }

# =========================================================
# ANALYZE
# =========================================================

@app.post("/analyze")
async def analyze_text(data: PatientData):

    encoded = tokenizer(
        data.text,
        max_length=64,
        truncation=True,
        padding='max_length',
        return_tensors='np'
    )

    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)

    # =====================================================
    # INPUT
    # =====================================================

    interpreter.set_tensor(
        input_details[0]['index'],
        input_ids
    )

    interpreter.set_tensor(
        input_details[1]['index'],
        attention_mask
    )

    # =====================================================
    # RUN
    # =====================================================

    interpreter.invoke()

    # =====================================================
    # OUTPUTS
    # =====================================================

    outputs = [
        interpreter.get_tensor(o['index'])
        for o in output_details
    ]

    stage_output = outputs[0]
    emotion_output = outputs[1]

    stage_idx = int(np.argmax(stage_output[0]))
    emotion_idx = int(np.argmax(emotion_output[0]))

    final_stage = stages[stage_idx]
    final_emotion = emotions[emotion_idx]

    # =====================================================
    # FIRESTORE SAVE
    # =====================================================

    try:

        db.collection("analyses").add({
            "userId": data.userId,
            "inputText": data.text,
            "stage": final_stage,
            "emotion": final_emotion,
            "timestamp": datetime.utcnow()
        })

    except Exception as e:
        print(f"Firestore hatası: {e}")

    # =====================================================
    # RESPONSE
    # =====================================================

    return {
        "stage": final_stage,
        "emotion": final_emotion
    }

# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )