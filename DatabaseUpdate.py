import os
import sqlite3
import torch
import numpy as np
from transformers import DPRContextEncoder, AutoTokenizer as DPRTokenizer

# DPR modelini ve tokenizer'ını yükleme
model_name_dpr = "facebook/dpr-ctx_encoder-single-nq-base"
tokenizer_dpr = DPRTokenizer.from_pretrained(model_name_dpr)
model_dpr = DPRContextEncoder.from_pretrained(model_name_dpr).to("cuda")

# Veritabanı bağlantısını oluşturma
def connect_db():
    return sqlite3.connect('Database.db')

# chunks tablosunu oluşturma
def init_chunks_table():
    conn = connect_db()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Chunks TEXT NOT NULL,
        DPRID BLOB NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Metin dosyasındaki paragrafları okuma ve boş satırlara göre ayırma
def read_paragraphs_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    return paragraphs

# DPR embedding'lerini hesaplama
def calculate_embeddings(paragraphs):
    embeddings = []
    for paragraph in paragraphs:
        inputs = tokenizer_dpr(paragraph, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            embeds = model_dpr(**inputs).pooler_output.cpu().numpy()
        embeddings.append(embeds)
    return embeddings

# Paragrafları ve embedding'lerini veritabanına ekleme
def insert_chunks_into_db(paragraphs, embeddings):
    conn = connect_db()
    c = conn.cursor()
    
    for paragraph, embedding in zip(paragraphs, embeddings):
        c.execute('''
        INSERT INTO chunks (Chunks, DPRID)
        VALUES (?, ?)
        ''', (paragraph, sqlite3.Binary(embedding.tobytes())))
    
    conn.commit()
    conn.close()

# Ana işlem akışı
def process_file_to_db(file_path):
    init_chunks_table()  # Tabloyu oluştur
    paragraphs = read_paragraphs_from_file(file_path)  # Metin dosyasını işle
    embeddings = calculate_embeddings(paragraphs)  # Embedding'leri hesapla
    insert_chunks_into_db(paragraphs, embeddings)  # Veritabanına ekle
    print("Chunks ve embedding'ler başarıyla veritabanına eklendi.")

# Kullanım
if __name__ == "__main__":
    text_file_path = "datas.txt"  # İşlenecek metin dosyasının yolu
    if os.path.exists(text_file_path):
        process_file_to_db(text_file_path)
    else:
        print(f"{text_file_path} bulunamadı. Lütfen dosya yolunu kontrol edin.")
