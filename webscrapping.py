import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM

# Wikipedia sayfasına istek gönder
url = "https://tr.wikipedia.org/wiki/Merkez%C3%AE_i%C5%9Flem_birimi"
response = requests.get(url)

# Sayfanın HTML içeriğini BeautifulSoup ile analiz et
soup = BeautifulSoup(response.content, "html.parser")
content = soup.find(id="mw-content-text")

# Tüm başlıkları ve paragrafları bul ve listeye ekle
data = []
for element in content.find_all(['h2', 'h3', 'h4', 'p']):
    if element.name in ['h2', 'h3', 'h4']:
        title = element.get_text().strip()
        data.append(f"\n\n### {title} ###\n")
    elif element.name == 'p':
        paragraph = element.get_text().strip()
        data.append(paragraph)

# Tüm verileri tek bir metin haline getir
full_text = "\n".join(data)

# Metni LLaMA modeli ile işlemek
print("\nLLaMA modeli başlatılıyor...")

# Tokenizer ve model yükleme (örneğin bir LLaMA türevi Hugging Face modeli)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer= AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_auth_token="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"  #huggingface token
)


# Modelden çıktı almak için veriyi tokenize et
inputs = tokenizer(full_text, return_tensors="pt", max_length=2048, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=150)

# Model çıktısını çöz ve göster
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nLLaMA Model Output:\n")
print(decoded_output)

# Sonuçları bir dosyaya kaydet
with open("sayfa_icerik.txt", "w", encoding="utf-8") as file:
    file.write(full_text)

print("\nVeri 'sayfa_icerik.txt' dosyasına başarıyla kaydedildi!")
