from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time

def scrape_website(url):
    # Chrome tarayıcıyı headless (arka planda) modda açma
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")  # Tarayıcıyı arka planda çalıştırır
    driver = webdriver.Chrome(options=chrome_options)
    
    driver.get(url)
    
    # Sayfanın tamamen yüklenmesini beklemek için document.readyState kontrolü
    WebDriverWait(driver, 30).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )

    # Sayfa kaynağını al ve BeautifulSoup ile analiz et
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Verileri toplamak için bir liste oluştur
    data = []
    
    # Sayfadaki tüm başlık ve paragrafları sırasıyla bul
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'figcaption']):
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            title = element.get_text().strip()
            if title:
                data.append(f"\n\n### {title} ###\n")
        elif element.name == 'p':
            paragraph = element.get_text().strip()
            if paragraph:
                data.append(paragraph)
        elif element.name == 'figcaption':
            caption = element.get_text().strip()
            if caption:
                data.append(f"\n\n**Figcaption:** {caption}\n")

    # Tarayıcıyı kapat
    driver.quit()
    
    # Tüm verileri birleştir
    full_text = "\n".join(data)
    
    # Veriyi bir dosyaya kaydet
    with open("sayfa_icerik.txt", "w", encoding="utf-8") as file:
        file.write(full_text)

    print("\nVeri 'sayfa_icerik.txt' dosyasına başarıyla kaydedildi!")

# URL girin
url = "https://hasdata.com/blog/how-to-scrape-dynamic-content-in-python"
scrape_website(url)
