import sqlite3

# Veritabanını oluşturuyoruz
conn = sqlite3.connect("islemci_bilgileri.db")
cursor = conn.cursor()

# TXT dosyasını okuyoruz
with open("veri.txt", "r", encoding="utf-8") as file:
    content = file.readlines()

# Tabloları oluşturmak için SQL komutları
tables = [
    ("islemci_nedir", "id INTEGER PRIMARY KEY AUTOINCREMENT, tanim TEXT"),
    ("islemci_ozellikleri", "id INTEGER PRIMARY KEY AUTOINCREMENT, ozellik TEXT"),
    ("islemci_turleri", "id INTEGER PRIMARY KEY AUTOINCREMENT, tur TEXT"),
    ("islemci_temel_islevi", "id INTEGER PRIMARY KEY AUTOINCREMENT, islev TEXT"),
    ("islemci_nasil_calisir", "id INTEGER PRIMARY KEY AUTOINCREMENT, calisma_prensibi TEXT"),
    ("islemci_performans_etkisi", "id INTEGER PRIMARY KEY AUTOINCREMENT, etki TEXT"),
    ("islemci_frekansi", "id INTEGER PRIMARY KEY AUTOINCREMENT, frekans TEXT"),
    ("islemci_alinirken_dikkat", "id INTEGER PRIMARY KEY AUTOINCREMENT, oneri TEXT"),
    ("islemci_tarihi", "id INTEGER PRIMARY KEY AUTOINCREMENT, tarih TEXT"),
    ("islemci_bilesenleri", "id INTEGER PRIMARY KEY AUTOINCREMENT, bilesen TEXT"),
    ("saat_vurus_sikligi", "id INTEGER PRIMARY KEY AUTOINCREMENT, siklik TEXT"),
    ("paralellik", "id INTEGER PRIMARY KEY AUTOINCREMENT, paralellik TEXT"),
    ("islemcide_nm", "id INTEGER PRIMARY KEY AUTOINCREMENT, nm_aciklama TEXT")
]

# Tabloları oluşturma
for table_name, columns in tables:
    try:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    except Exception as e:
        print(f"Hata oluştu: {e}")

# TXT dosyasındaki satır aralıklarına göre verileri eklemek için eşleme
araliklar = {
    "islemci_nedir": [(1, 3), (43, 45), (128, 129), (190, 193), (251, 252), (256, 257), (300, 303)],
    "islemci_ozellikleri": [(44, 51), (115, 123), (173, 185), (256, 277)],
    "islemci_turleri": [(26, 35), (95, 101)],
    "islemci_temel_islevi": [(4, 13), (84, 90)],
    "islemci_nasil_calisir": [(15, 24), (53, 61), (158, 170), (208, 218)],
    "islemci_performans_etkisi": [(37, 38), (72, 74), (241, 246)],
    "islemci_frekansi": [(72, 74), (111, 112), (279, 282)],
    "islemci_alinirken_dikkat": [(76, 82), (103, 109)],
    "islemci_tarihi": [(63, 70), (128, 129), (192, 206), (319, 327)],
    "islemci_bilesenleri": [(141, 157)],
    "saat_vurus_sikligi": [(227, 231)],
    "paralellik": [(233, 239)],
    "islemcide_nm": [(284, 287)]
}

# Verileri eklemek için döngü
try:
    for tablo, ranges in araliklar.items():
        # Hangi tablo için hangi sütunun ekleneceğini belirlemek
        sutun = next((col.split()[0] for table, col in tables if table == tablo), "tanim")

        for start, end in ranges:
            # Satır aralığındaki verileri birleştiriyoruz
            if start - 1 < 0 or end > len(content):
                print(f"Geçersiz satır aralığı: {start}-{end} için {tablo}")
                continue
            
            # Eğer geçerli bir aralıksa, seçilen satırları alıyoruz
            selected_lines = content[start - 1:end]  # Python'da index 0'dan başlar, bu yüzden -1 yapıyoruz
            combined_text = " ".join([line.strip() for line in selected_lines])
            
            # Veriyi tabloya ekliyoruz
            cursor.execute(f"INSERT INTO {tablo} ({sutun}) VALUES (?)", (combined_text,))
except Exception as e:
    print(f"Hata oluştu: {e}")

# Değişiklikleri kaydediyoruz ve bağlantıyı kapatıyoruz
conn.commit()
conn.close()

print("Tüm tablolar başarıyla oluşturuldu ve veriler eklendi.")
