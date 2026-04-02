import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLO modelini yükle (ilk çalıştırmada otomatik indirir)
model = YOLO("yolov8n.pt")

# Kamera aç
cap = cv2.VideoCapture(0)

# FPS hesaplama
prev_time = 0

def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Küçük gürültüleri filtrele
        if area < 1000:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            shape = "Ucgen"
        elif len(approx) == 4:
            shape = "Kare/Dikdortgen"
        elif len(approx) > 6:
            shape = "Daire"
        else:
            shape = "Bilinmeyen"

        cv2.drawContours(frame, [approx], 0, (0, 255, 255), 2)
        cv2.putText(frame, shape, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO NESNE TESPİTİ
    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            # Kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Etiket yaz
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # ŞEKİL TESPİTİ (OpenCV)
    frame = detect_shapes(frame)

    # FPS hesapla
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    # Ekranda göster
    cv2.imshow("YOLO + Sekil Tanima", frame)

    # q ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
