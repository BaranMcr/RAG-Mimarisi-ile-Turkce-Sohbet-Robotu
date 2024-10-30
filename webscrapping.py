import requests
from bs4 import BeautifulSoup

# 1. Wikipedia sayfasına istek gönderen kod kısmı
url = "https://tr.wikipedia.org/wiki/Merkez%C3%AE_i%C5%9Flem_birimi"
response = requests.get(url)

# 2. Sayfanın HTML içeriğini parse ederek okunabilir ve yazıları çekebilir hale getiriyoruz
soup = BeautifulSoup(response.content, "html.parser")

# 3. Wikipedia da başlık ve paragrafların tutulduğu id nin ismi mv-content-text tir.
content = soup.find(id="mw-content-text")

data = []  # Başlık ve paragrafları sıra ile eklemek için oluşturulan liste

# For döngüsü içinde tüm başlık ve paragrafları data listesine ekleyen kod
for element in content.find_all(['h2', 'h3', 'h4', 'p']):
    # Başlıkları alan kısım (h2,h3,h4 sırasıyla birbirlerinin alt başlıklarıdır).
    if element.name in ['h2', 'h3', 'h4']:
        title = element.get_text().strip()
        data.append(f"\n\n### {title} ###\n")
    # Paragraf olup olmadığına bakmak için 'p' kullanılır.
    elif element.name == 'p':
        paragraph = element.get_text().strip()
        data.append(paragraph)

# data listesindeki yazıları birleştirip full_text adlı string dosyasına aktar
full_text = "\n".join(data)

# text dosyası oluşturup yazma
with open("sayfa_icerik.txt", "w", encoding="utf-8") as file:
    file.write(full_text)

print("\nVeri 'sayfa_icerik.txt' dosyasına başarıyla kaydedildi!")