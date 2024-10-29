import requests
from bs4 import BeautifulSoup

# 1. Wikipedia sayfasına istek gönder
url = "https://tr.wikipedia.org/wiki/Merkez%C3%AE_i%C5%9Flem_birimi"
response = requests.get(url)

# 2. Sayfanın HTML içeriğini parse et
soup = BeautifulSoup(response.content, "html.parser")

# 3. Ana içerik bölümünü bul (ID: 'mw-content-text')
content = soup.find(id="mw-content-text")

# 4. Paragrafları dosyaya yaz
with open("paragraflar.txt", "w", encoding="utf-8") as file:
    for paragraph in content.find_all("p"):
        # Paragrafları dosyaya yazdır 
        file.write(paragraph.get_text() + "\n\n")

print("Paragraflar başarıyla 'paragraflar.txt' dosyasına yazıldı!")