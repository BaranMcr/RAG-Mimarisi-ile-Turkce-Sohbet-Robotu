import sqlite3
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRContextEncoder, AutoTokenizer as DPRTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Flask uygulamasını başlatma
app = Flask(__name__)

# Llm modeli ve tokenizer'ını yükleme
llm_model_name = "asafaya/kanarya-2b"
tokenizer_llm = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to("cuda")

# PAD token'ı ayarlama
if tokenizer_llm.pad_token_id is None:
    tokenizer_llm.pad_token = tokenizer_llm.eos_token

# DPR modelini ve tokenizer'ını yükleme
model_name_dpr = "facebook/dpr-ctx_encoder-single-nq-base"
tokenizer_dpr = DPRTokenizer.from_pretrained(model_name_dpr)
model_dpr = DPRContextEncoder.from_pretrained(model_name_dpr).to("cuda")

# Veritabanından paragraf ve embedding'leri yükleme
def load_chunks_from_db():
    conn = sqlite3.connect("Database.db")
    c = conn.cursor()
    c.execute("SELECT Chunks, DPRID FROM chunks")
    rows = c.fetchall()
    conn.close()
    
    paragraphs = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    return paragraphs, np.vstack(embeddings)

# En yakın bağlamı bulma
def retrieve(query, paragraphs, paragraph_embeds):
    # Sorgunun embedding'ini hesaplama
    inputs = tokenizer_dpr(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        query_embeds = model_dpr(**inputs).pooler_output.cpu().numpy()
    
    # Cosine similarity hesaplama
    similarities = cosine_similarity(query_embeds, paragraph_embeds)
    best_index = similarities.argsort()[0][-1]  # En yüksek skorun indeksini al
    return paragraphs[best_index]  # Sadece en yakın bağlamı döndür

# Llm modeli ile cevap üretme
def generate_answer(input_text, context):
    full_input = (
        f"Bağlam: {context}\n"
        f"Soru: {input_text}\n"
        f"Lütfen sadece kısa ve açık bir cevap ver."
    )
    inputs = tokenizer_llm(full_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer_llm.eos_token_id,
            pad_token_id=tokenizer_llm.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer_llm.decode(output[0], skip_special_tokens=True)

# Flask rota: Ana Sayfa
@app.route("/", methods=["GET", "POST"])
def home():
    # Veritabanındaki verileri yükleme
    paragraphs, paragraph_embeds = load_chunks_from_db()
    
    if request.method == "POST":
        user_input = request.form.get("query")  # Kullanıcıdan gelen sorguyu al
        context = retrieve(user_input, paragraphs, paragraph_embeds)  # En yakın bağlamı al
        
        print("Seçilen bağlam: " + context)  # Bağlamı yazdırmak

        response = generate_answer(user_input, context)  # Model yanıtını üret
        
        # "Cevap:" kelimesinden sonrası alınır ve son noktada kesilir
        if "Cevap:" in response:
            answer_start = response.find("Cevap:") + len("Cevap:")  # "Cevap:" kelimesinin sonrasını bul
            answer = response[answer_start:].strip()  # "Cevap:" sonrası kısmı al
            if answer.endswith("."):  # Son karakter nokta mı kontrol et
                answer = answer  # Cevap zaten noktada bitiyorsa olduğu gibi bırak
            else:
                answer = answer.rsplit(".", 1)[0] + "."  # Son noktadan kes ve noktayı ekle
        else:
            answer = "Üzgünüm, bu soruya uygun bir cevap bulunamadı."

        return render_template("index.html", user_input=user_input, answer=answer)
    return render_template("index.html")

# Flask uygulamasını çalıştırma
if __name__ == "__main__":
    app.run(debug=False)
