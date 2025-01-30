import sqlite3
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRContextEncoder, AutoTokenizer as DPRTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata
import string

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

# Kelime temizleme fonksiyonu
def clean_word(word):
    word = word.lower()
    word = unicodedata.normalize("NFKD", word).encode("ASCII", "ignore").decode("utf-8")
    word = word.strip(string.punctuation)
    return word

# En yakın bağlamı bulma (benzerlik oranı 0.5'ten düşük olanları filtreler)
def retrieve(query, paragraphs, paragraph_embeds):
    inputs = tokenizer_dpr(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        query_embeds = model_dpr(**inputs).pooler_output.cpu().numpy()
    
    similarities = cosine_similarity(query_embeds, paragraph_embeds)
    sorted_indices = similarities.argsort()[0][::-1]
    
    top_5_indices = sorted_indices[:5]
    filtered_paragraphs = []
    filtered_similarities = []
    
    for i in top_5_indices:
        if similarities[0][i] >= 0.5:
            filtered_paragraphs.append(paragraphs[i])
            filtered_similarities.append(similarities[0][i])
    
    print("\nEn Yakın 5 Bağlam (0.5 üstü benzerlik):")
    for i, (paragraph, similarity) in enumerate(zip(filtered_paragraphs, filtered_similarities)):
        print(f"{i+1}. {paragraph}\n  Benzerlik Değeri: {similarity}")
    
    return filtered_paragraphs, filtered_similarities

# En iyi eşleşen bağlamı bulma
def find_best_match(query, top_paragraphs):
    if not top_paragraphs:
        return -1, set()
    
    query_words = [clean_word(word) for word in query.split()]
    best_match_index = -1
    max_matching_words = 0
    best_common_words = set()
    
    for i, paragraph in enumerate(top_paragraphs):
        paragraph_words = [clean_word(word) for word in paragraph.split()]
        common_words = set(query_words).intersection(paragraph_words)
        
        if len(common_words) > max_matching_words:
            max_matching_words = len(common_words)
            best_match_index = i
            best_common_words = common_words
    
    print(f"\nEn İyi Eşleşen Bağlam: {top_paragraphs[best_match_index]}")
    print(f"Eşleşen Kelimeler: {', '.join(best_common_words)}")
    
    return best_match_index, best_common_words

# Llm modeli ile cevap üretme
def generate_answer(input_text, context):
    full_input = (
        f"Bağlam: {context}\n"
        f"Soru: {input_text}\n"
        f"Cevap:"
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
    response = tokenizer_llm.decode(output[0], skip_special_tokens=True)
    
    if "Cevap:" in response:
        answer_start = response.find("Cevap:") + len("Cevap:")
        answer = response[answer_start:].strip()
        if "." in answer:
            answer = answer.rsplit(".", 1)[0] + "."
    else:
        answer = "Üzgünüm, bu soruya uygun bir cevap bulunamadı."
    
    return answer

# Flask rota: Ana Sayfa
@app.route("/", methods=["GET", "POST"])
def home():
    paragraphs, paragraph_embeds = load_chunks_from_db()
    
    if request.method == "POST":
        user_input = request.form.get("query")
        best_paragraphs, best_similarities = retrieve(user_input, paragraphs, paragraph_embeds)
        best_match_index, best_common_words = find_best_match(user_input, best_paragraphs)
        
        context = best_paragraphs[best_match_index] if best_match_index != -1 else None
        print(f"\nEn Son Karar Verilen Bağlam: {context}")
        
        answer = generate_answer(user_input, context) if context else "Cevap: Üzgünüm, bu soruyu cevaplamak için yeterli bilgiye sahip değilim."
        
        return render_template("index.html", user_input=user_input, answer=answer)
    
    return render_template("index.html")

# Flask uygulamasını çalıştırma
if __name__ == "__main__":
    app.run(debug=False)
