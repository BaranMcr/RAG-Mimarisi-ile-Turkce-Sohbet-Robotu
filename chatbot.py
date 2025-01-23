import sqlite3
<<<<<<< HEAD
from flask import Flask, render_template, request
=======
>>>>>>> 932a972162a5e9229ac666dd68200d88a98646b8
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRContextEncoder, AutoTokenizer as DPRTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

<<<<<<< HEAD
# Flask uygulamasını başlatma
app = Flask(__name__)

# Llama Türkçe modeli ve tokenizer'ını yükleme
model_name_llama = "asafaya/kanarya-2b"
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama).to("cuda")

# PAD token'ı ayarlama
if tokenizer_llama.pad_token_id is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token

# DPR modelini ve tokenizer'ını yükleme
=======
# Llama modelini ve tokenizer'ını yükliyoruz
model_name_llama = "meta-llama/Llama-3.2-1B"
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)  # Tokenizer'ı Llama modeline uygun şekilde yüklüyoruz
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama).to("cuda")  # Modeli yükleyip GPU'ya taşıyoruz

# DPR modelini ve tokenizer'ını yüklüyoruz
>>>>>>> 932a972162a5e9229ac666dd68200d88a98646b8
model_name_dpr = "facebook/dpr-ctx_encoder-single-nq-base"
tokenizer_dpr = DPRTokenizer.from_pretrained(model_name_dpr)
model_dpr = DPRContextEncoder.from_pretrained(model_name_dpr).to("cuda")

<<<<<<< HEAD
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
=======
# Paragraflar.txt dosyasını okuyarak metinleri listeliyoruz
# Bu dosya, sistemin bilgi havuzu olarak kullanılacak
with open("veri.txt", "r", encoding="utf-8") as f:
    content = f.read()  # Dosyanın tamamını okuyoruz
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]  # Boş satırlara göre bölüyoruz ve temizliyoruz

# Verilerin doğru şekilde yüklendiğini kontrol et
if not paragraphs:
    print("Paragraflar yüklenemedi. Dosya boş veya hatalı.")
else:
    print(f"Toplam paragraf sayısı: {len(paragraphs)}")
>>>>>>> 932a972162a5e9229ac666dd68200d88a98646b8

# En iyi bağlamları bulma
def retrieve(query, paragraphs, paragraph_embeds, top_n=3):
    # Sorgunun embedding'ini hesaplama
    inputs = tokenizer_dpr(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        query_embeds = model_dpr(**inputs).pooler_output.cpu().numpy()
    
<<<<<<< HEAD
    # Cosine similarity hesaplama
=======
    # Paragrafların her biri için embedding hesaplıyoruz
    paragraph_embeds = []
    for paragraph in paragraphs:
        inputs = tokenizer_dpr(paragraph, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            # Embedding'i hesapla ve listeye ekle
            paragraph_embed = model_dpr(**inputs).pooler_output.cpu().numpy()
            paragraph_embeds.append(paragraph_embed)
    
    # Eğer paragraflar için embedding hesaplanamamışsa, bir mesaj yazdır ve işlem yapma
    if not paragraph_embeds:
        print("Hiçbir paragrafın embedding'i hesaplanamadı.")
        return None

    # Hem sorgu hem de paragrafların embedding'lerini 2D array'e dönüştürmemiz gerekiyor
    query_embeds = query_embeds.reshape(1, -1)  # Sorgu embedding'ini 2D array'e çeviriyoruz
    paragraph_embeds = np.vstack(paragraph_embeds)  # Paragrafların embedding'lerini 2D array'e döküyoruz

    # Cosine similarity kullanarak, sorguya en yakın paragrafı buluyoruz
>>>>>>> 932a972162a5e9229ac666dd68200d88a98646b8
    similarities = cosine_similarity(query_embeds, paragraph_embeds)
    best_indices = similarities.argsort()[0][-top_n:][::-1]
    return [paragraphs[i] for i in best_indices]

# Llama modeli ile cevap üretme
def generate_answer(input_text, context):
<<<<<<< HEAD
    full_input = (
        f"Bağlam: {context}\n"
        f"Soru: {input_text}\n"
        f"Lütfen sadece kısa ve açık bir cevap ver."
    )
    inputs = tokenizer_llama(full_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        output = model_llama.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer_llama.eos_token_id,
            pad_token_id=tokenizer_llama.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer_llama.decode(output[0], skip_special_tokens=True)
=======
    # Kullanıcıdan gelen soru ve ilgili bağlamı birleştiriyoruz
    full_input = f"User: {input_text}\nContext: {context}"
    # Kullanıcı inputunu token'ize ediyoruz
    inputs = tokenizer_llama(full_input, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    # Modelden cevabı üretiyoruz
    with torch.no_grad():  # Yine gradients'e ihtiyacımız yok
        output = model_llama.generate(**inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.3, no_repeat_ngram_size=3)
    
    # Cevap metnini çözümlüyoruz (token'lerden tekrar metne dönüştürüyoruz)
    response = tokenizer_llama.decode(output[0], skip_special_tokens=True)
    return response
>>>>>>> 932a972162a5e9229ac666dd68200d88a98646b8

# Flask rota: Ana Sayfa
@app.route("/", methods=["GET", "POST"])
def home():
    # Veritabanındaki verileri yükleme
    paragraphs, paragraph_embeds = load_chunks_from_db()
    
<<<<<<< HEAD
    if request.method == "POST":
        user_input = request.form.get("query")  # Kullanıcıdan gelen sorguyu al
        context = retrieve(user_input, paragraphs, paragraph_embeds)  # Sorguya uygun bağlamları al
        
        # Bağlamları birleştiriyoruz
        context_str = " ".join(context)
        print("Seçilen bağlam: " + context_str)  # Bağlamı yazdırmak
=======
    if context is None:
        return "Üzgünüm, uygun bir bağlam bulunamadı."

    print(f"Seçilen bağlam: {context}")
    
    # Generator fonksiyonu ile cevabı üretiyoruz
    response = generate_answer(input_text, context)
    return response
>>>>>>> 932a972162a5e9229ac666dd68200d88a98646b8

        response = generate_answer(user_input, context_str)  # Model yanıtını üret
        
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
