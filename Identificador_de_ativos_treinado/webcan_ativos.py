from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

# === CONFIGURAÇÃO DA CÂMERA ===
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# Ajusta resolução
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()
else:
    print("Câmera conectada.")

# === YOLO ===
#model = YOLO("yolov8n.pt")
#model = YOLO("runs/detect/train5/weights/best.pt")
model = YOLO("C:/Users/Usuario/Desktop/A3/modelagem e métodos/A3_só_necessário/Identificador_de_ativos_treinado/runs/detect/train/weights/best.pt")

track_history = defaultdict(lambda: [])
seguir = False
deixar_rastro = False

while True:
    success, img = cap.read()
    if not success:
        print("Falha ao capturar imagem.")
        break

    results = model.track(img, persist=True) if seguir else model(img)

    for result in results:
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], False, (230, 0, 0), 5)
            except:
                pass

    cv2.imshow("YOLOv8 - Webcam", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        print("Encerrando programa...")
        break

    if cv2.getWindowProperty("YOLOv8 - Webcam", cv2.WND_PROP_VISIBLE) < 1:
        print("Stream finalizada.")
        break

cap.release()
cv2.destroyAllWindows()
