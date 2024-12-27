import sqlite3
import torch
from transformers import AutoTokenizer, DPRContextEncoder, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Llama modelini ve tokenizer'ını yükliyoruz
model_name_llama = "meta-llama/Llama-3.2-1B"
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)  # Tokenizer'ı Llama modeline uygun şekilde yüklüyoruz
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama).to("cuda")  # Modeli yükleyip GPU'ya taşıyoruz

# DPR modelini ve tokenizer'ını yüklüyoruz
model_name_dpr = "facebook/dpr-ctx_encoder-single-nq-base"
tokenizer_dpr = AutoTokenizer.from_pretrained(model_name_dpr)  # Tokenizer'ı DPR modeline uygun şekilde yüklüyoruz
model_dpr = DPRContextEncoder.from_pretrained(model_name_dpr).to("cuda")  # Modeli yükleyip GPU'ya taşıyoruz

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

# Retriever fonksiyonu: Kullanıcı sorgusu ile en uygun paragrafı buluyoruz
def retrieve(query):
    # Kullanıcıdan gelen sorguyu token'ize ediyoruz (metni modele uygun hale getiriyoruz)
    inputs = tokenizer_dpr(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():  # Gradients'e ihtiyacımız olmadığı için modelin hesaplama adımlarını hızlandırıyoruz
        query_embeds = model_dpr(**inputs).pooler_output.cpu().numpy()  # Query için embedding'i alıyoruz
    
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
    similarities = cosine_similarity(query_embeds, paragraph_embeds)
    best_idx = np.argmax(similarities)  # En yüksek benzerliğe sahip paragrafın index'ini alıyoruz
    return paragraphs[best_idx]  # En uygun paragrafı döndürüyoruz

# Generator fonksiyonu: Llama modelini kullanarak cevap üretiyoruz
def generate_answer(input_text, context):
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

# Ana chatbot fonksiyonu
def chatbot(input_text):
    # Retriever fonksiyonu ile bağlamı (context) alıyoruz
    context = retrieve(input_text)
    
    if context is None:
        return "Üzgünüm, uygun bir bağlam bulunamadı."

    print(f"Seçilen bağlam: {context}")
    
    # Generator fonksiyonu ile cevabı üretiyoruz
    response = generate_answer(input_text, context)
    return response

# Sohbet başlatma: Kullanıcıdan gelen mesajları alıyoruz ve chatbot'un cevabını veriyoruz
print("Chatbot başlatıldı. Sorularınızı yazabilirsiniz!")

# Sonsuz döngüde kullanıcıyla etkileşim kuruyoruz
while True:
    user_input = input("Siz: ")  # Kullanıcıdan gelen input
    if user_input.lower() in ["exit", "quit", "bye"]:  # Çıkmak için komutlar
        print("Chatbot sonlandırılıyor...")
        break  # Döngüyü kırıyoruz
    
    # Chatbot'tan cevabı alıyoruz ve ekrana yazdırıyoruz
    response = chatbot(user_input)
    print(f"Chatbot: {response}")
