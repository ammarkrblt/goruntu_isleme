import cv2
import numpy as np

# Kamerayı aç
video_capture = cv2.VideoCapture(0)

# Bir kareyi al
ret, frame = video_capture.read()

# Kamerayı kapat
video_capture.release()

if ret:
    # Görüntüyü gri tonlamaya çevir
    gri_tonlu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Görüntüyü bulanıklaştır
    bulanik = cv2.GaussianBlur(gri_tonlu, (5, 5), 0)

    # Kenarları tespit et
    kenarlar = cv2.Canny(bulanik, 50, 150)

    # Morfolojik işlemler uygula
    kernel = np.ones((5, 5), np.uint8)
    genisletilmis = cv2.dilate(kenarlar, kernel, iterations=1)

    # Kontur tespiti yap
    konturlar, _ = cv2.findContours(genisletilmis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pirinç tanelerini say
    pirinc_sayisi = 0
    for kontur in konturlar:
        alan = cv2.contourArea(kontur)
        if alan > 100:  # Eşik değeri ile oynayarak pirinç tanelerini belirleyebilirsiniz
            pirinc_sayisi += 1
            cv2.drawContours(frame, [kontur], -1, (0, 255, 0), 2)

    print(f"Pirinç Taneleri Sayısı: {pirinc_sayisi}")

    # Sonucu göster
    cv2.imshow('Pirinç Sayım', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Kamera açılamadı.")
