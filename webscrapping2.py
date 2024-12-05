from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup

def scrape_website(url):
    # Chrome tarayıcıyı arka planda aç       ###Firefox tarayıcı için ekleme yapılacak
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    
    driver.get(url)
    
    # Sayfanın tamamen yüklenmesini beklemek için document.readyState kontrolü
    WebDriverWait(driver, 30).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )

    # Sayfa kaynağını al ve BeautifulSoup ile Parse et
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Ana içeriği bul (Vikipedi için "mw-content-text" id'si)
    main_content = soup.find(id="mw-content-text")
    
    # Verilerin toplanacağı liste
    data = []
    
    # Ana içerik içinde başlık, paragraf ve listeleri sırasıyla bul
    if main_content:
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'figcaption', 'ul', 'ol']):
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
            elif element.name in ['ul', 'ol']:  # Liste elemanları
                for li in element.find_all('li'):
                    list_item = li.get_text().strip()
                    if list_item:
                        data.append(f"- {list_item}")
                data.append("\n")

    driver.quit()
    
    # Tüm verileri birleştir
    full_text = "\n".join(data)
    
    # Veriyi bir dosyaya kaydet
    with open("sayfa_icerik.txt", "w", encoding="utf-8") as file:
        file.write(full_text)

    print("\nVeri 'sayfa_icerik.txt' dosyasına başarıyla kaydedildi!")

# Veri çekilecek URL
url = "https://tr.wikipedia.org/wiki/Balkanlar"
scrape_website(url)
