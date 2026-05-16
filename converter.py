import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
from transformers import TFAutoModel

# =========================================================
# CONFIG (Hata mesajına göre 64 olarak güncellendi)
# =========================================================
model_name = "ytu-ce-cosmos/turkish-small-bert-uncased"
num_stages = 5     
num_emotions = 6   
max_length = 64    # Hata mesajındaki shape=(1, 64) ile eşitlendi

# =========================================================
# MODEL MİMARİSİ
# =========================================================
class ModelSenaryo2(tf.keras.Model):
    def __init__(self, bert_backbone, num_stages, num_emotions):
        super().__init__()
        self.bert = bert_backbone
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.stage_head = tf.keras.layers.Dense(num_stages, activation='softmax', name='asama')
        self.emotion_head = tf.keras.layers.Dense(num_emotions, activation='softmax', name='duygu')

    # input_signature kısmını liste [ ... ] yapısına çevirerek hatayı çözüyoruz
    @tf.function(input_signature=[
        [tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name="input_ids"),
         tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name="attention_mask")]
    ])
    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs[0], inputs[1]
        
        cls_token = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        ).pooler_output
        
        x = self.dropout(cls_token, training=training)
        return {
            'asama': self.stage_head(x),
            'duygu': self.emotion_head(x)
        }

# =========================================================
# MODELİ YÜKLEME SÜRECİ
# =========================================================
print("Backbone yükleniyor...")
bert_backbone = TFAutoModel.from_pretrained(model_name)

print("Model oluşturuluyor...")
model = ModelSenaryo2(bert_backbone, num_stages, num_emotions)

print("Weights yükleniyor...")
model.load_weights("house_md_model.h5", by_name=True, skip_mismatch=True)

# =========================================================
# TFLITE DÖNÜŞÜMÜ
# =========================================================
print("TFLite dönüşümü başlıyor...")

# Burada da concrete_func imzasını liste yapısına çektik
run_model = tf.function(lambda x: model(x, training=False))
concrete_func = run_model.get_concrete_function([
    tf.TensorSpec(shape=[1, max_length], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[1, max_length], dtype=tf.int32, name="attention_mask")
])

# Dönüştürücü kurulumu
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.target_spec.supported_types = [tf.float16]

# Dönüştür ve Kaydet
tflite_model = converter.convert()

with open("house_md_small.tflite", "wb") as f:
    f.write(tflite_model)

print("🚀 Muhteşem! TFLite Modeli Başarıyla Oluşturuldu: house_md_small.tflite")