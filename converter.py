import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from transformers import TFBertModel, BertConfig

# =========================================================
# CONFIG
# =========================================================
model_name = "ytu-ce-cosmos/turkish-small-bert-uncased"
num_stages = 5     # Yeni mapping sonrası Aşama sayısı: 5
num_emotions = 5   # Yeni mapping sonrası Duygu sayısı: 5
max_length = 64    

# =========================================================
# MODEL MİMARİSİNİ YENİDEN KURMA (Functional API)
# =========================================================
print("Mimari kuruluyor...")

# Transformers yapısı
config = BertConfig.from_pretrained(model_name)
bert_backbone = TFBertModel(config)

# Girişler
input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

# Bağlantılar
bert_outputs = bert_backbone(input_ids=input_ids, attention_mask=attention_mask)
cls_token = bert_outputs.pooler_output
dropout = tf.keras.layers.Dropout(0.3)(cls_token)

# Çıkışlar
stage_head = tf.keras.layers.Dense(num_stages, activation='softmax', name='asama')(dropout)
emotion_head = tf.keras.layers.Dense(num_emotions, activation='softmax', name='duygu')(dropout)

# Modeli Birleştirme
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[stage_head, emotion_head])

# =========================================================
# AĞIRLIKLARI YÜKLEME
# =========================================================
print("Eğitilmiş ağırlıklar yükleniyor (senaryo2.h5)...")
# Eğittiğimiz Functional modelin ağırlıklarını doğrudan içeri alıyoruz
model.load_weights("senaryo2.h5")

# =========================================================
# TFLITE DÖNÜŞÜMÜ (Artık çok daha basit!)
# =========================================================
print("TFLite dönüşümü başlıyor...")

# Doğrudan Keras modelinden dönüştürücü oluştur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
# İsteğe bağlı: Modeli daha da küçültmek için float16 optimizasyonu
converter.target_spec.supported_types = [tf.float16]

# Dönüştür
tflite_model = converter.convert()

# Kaydet
with open("house_md_senaryo2.tflite", "wb") as f:
    f.write(tflite_model)

print("🚀 Muhteşem! TFLite Modeli Başarıyla Oluşturuldu: house_md_senaryo2.tflite")