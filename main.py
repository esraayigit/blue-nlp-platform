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
# TOKENIZER VE ETİKETLER (MAPPINGS)
# =========================================================

model_name = "ytu-ce-cosmos/turkish-small-bert-uncased"
print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer hazır.")

# YENİ MODELİN ÇIKIŞ ETİKETLERİ (Colab'den alındı)
intents = ['Açıklama', 'Hipotez/Fikir', 'Soru', 'Tanı/Test', 'Talimat/Tedavi', 'İtiraz']
stages = ['Hipotez/Değerlendirme', 'Test/Tanı', 'Diğer', 'Tedavi/Müdahale']
emotions = ['Nötr/Ciddi', 'Alaycı', 'Empatik/Düşünceli', 'Kaygılı/Korkmuş', 'Kararlı/Emin', 'Şaşkın']

# =========================================================
# TFLITE MODEL
# =========================================================

print("TFLite model yükleniyor...")
# BURAYA YENİ TFLITE DOSYASININ ADINI YAZIN
interpreter = tf.lite.Interpreter(model_path="house_md_v2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model hazır.")

# TFLite indeks eşleşmeleri (Colab Hücre 13 çıktısına göre)
# [1 6] -> Intent
INDEX_INTENT = next(o['index'] for o in output_details if o['shape'][1] == 6 and 'StatefulPartitionedCall:0' in o['name'])
# [1 6] -> Emotion
INDEX_EMOTION = next(o['index'] for o in output_details if o['shape'][1] == 6 and 'StatefulPartitionedCall:1' in o['name'])
# [1 1] -> Sarcasm
INDEX_SARCASM = next(o['index'] for o in output_details if o['shape'][1] == 1)
# [1 4] -> Stage
INDEX_STAGE = next(o['index'] for o in output_details if o['shape'][1] == 4)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="House MD - NLP Analizi (Multi-Task)")

class PredictRequest(BaseModel):
    userId: str
    text: str

@app.post("/predict")
def predict(data: PredictRequest):

    # =====================================================
    # PREPROCESSING
    # =====================================================
    tokens = tokenizer(
        data.text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    # Tensorları TFLite formatına uyarlama (int32)
    input_ids = np.array(tokens["input_ids"], dtype=np.int32)
    attention_mask = np.array(tokens["attention_mask"], dtype=np.int32)

    # =====================================================
    # INFERENCE (TAHMİN)
    # =====================================================
    interpreter.set_tensor(input_details[0]['index'], attention_mask)
    interpreter.set_tensor(input_details[1]['index'], input_ids)

    interpreter.invoke()

    # =====================================================
    # OUTPUTS (SONUÇLARI AYIKLAMA)
    # =====================================================
    
    out_sarcasm = interpreter.get_tensor(INDEX_SARCASM)[0][0]
    out_intent = interpreter.get_tensor(INDEX_INTENT)[0]
    out_stage = interpreter.get_tensor(INDEX_STAGE)[0]
    out_emotion = interpreter.get_tensor(INDEX_EMOTION)[0]

    # Değerleri anlamlandırma
    final_sarcasm = "Evet" if out_sarcasm > 0.5 else "Hayır"
    final_intent = intents[np.argmax(out_intent)]
    final_stage = stages[np.argmax(out_stage)]
    final_emotion = emotions[np.argmax(out_emotion)]

    # =====================================================
    # FIRESTORE SAVE
    # =====================================================
    try:
        db.collection("analyses").add({
            "userId": data.userId,
            "inputText": data.text,
            "sarcasm": final_sarcasm,
            "intent": final_intent,
            "stage": final_stage,
            "emotion": final_emotion,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        print(f"Firestore hatası: {e}")

    # =====================================================
    # RESPONSE (FLUTTER'A DÖNÜŞ)
    # =====================================================
    return {
        "sarcasm": final_sarcasm,
        "intent": final_intent,
        "stage": final_stage,
        "emotion": final_emotion
    }

# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
