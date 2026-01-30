# vision_app.py
import cv2
import numpy as np
from collections import defaultdict
import traceback

# --------------------------------------------------------
# CONFIGURAÇÃO AUTOMÁTICA DO MODELO (CAMINHO DO SEU CÓDIGO REAL)
# --------------------------------------------------------
MODEL_PATH = r"C:/Users/Usuario/Desktop/A3/modelagem e métodos/A3_só_necessário/Identificador_de_ativos_treinado/runs/detect/train/weights/best.pt"

# --------------------------------------------------------
# AssetDetectionModel – MELHORADO E COMPATÍVEL COM SEU CÓDIGO REAL
# --------------------------------------------------------
class AssetDetectionModel:
    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("Erro importando ultralytics:", e)
            raise

        print(f"Carregando modelo YOLO: {model_path}")

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Erro ao carregar pesos em '{model_path}': {e}")
            raise

        self.track_history = defaultdict(list)

    def infer(self, frame, seguir=False, deixar_rastro=False):
        try:
            results = self.model.track(frame, persist=True) if seguir else self.model(frame)
        except Exception as e:
            print("Erro na inferência YOLO:", e)
            traceback.print_exc()
            return frame, None

        img_plot = frame.copy()

        try:
            for result in results:
                # Plot das detecções (igual ao seu código)
                try:
                    img_plot = result.plot()
                except:
                    pass

                # Adiciona rastro no modo tracking
                if seguir and deixar_rastro:
                    try:
                        boxes_attr = getattr(result, "boxes", None)
                        if boxes_attr is None:
                            continue

                        # Suporta xywh ou xyxy dependendo da versão
                        if hasattr(boxes_attr, "xywh"):
                            boxes = boxes_attr.xywh.cpu().numpy()
                        elif hasattr(boxes_attr, "xyxy"):
                            boxes = boxes_attr.xyxy.cpu().numpy()
                        else:
                            boxes = None

                        ids_attr = getattr(boxes_attr, "id", None)
                        if ids_attr is not None:
                            try:
                                track_ids = ids_attr.int().cpu().tolist()
                            except:
                                track_ids = ids_attr.cpu().tolist()
                        else:
                            track_ids = []

                        if boxes is not None and len(track_ids) == len(boxes):
                            for box, track_id in zip(boxes, track_ids):
                                x = float(box[0])
                                y = float(box[1])
                                self.track_history[track_id].append((x, y))
                                if len(self.track_history[track_id]) > 30:
                                    self.track_history[track_id].pop(0)

                    except Exception:
                        pass

        except Exception:
            traceback.print_exc()

        return img_plot, results

    def get_tracks(self):
        return dict(self.track_history)


# --------------------------------------------------------
# NavigationModel – placeholder
# --------------------------------------------------------
class NavigationModel:
    def analyzeTerrain(self, frame):
        return {"slope": 0.0, "roughness": 0.0}

    def detectObstacles(self, frame):
        return []


# --------------------------------------------------------
# VisionSystem – Integrando IA de ativos + IA de navegação
# --------------------------------------------------------
class VisionSystem:
    def __init__(self, asset_model: AssetDetectionModel, nav_model: NavigationModel):
        self.assetModel = asset_model
        self.navModel = nav_model

    def detectAssets(self, frame, seguir=False, deixar_rastro=False):
        return self.assetModel.infer(frame, seguir=seguir, deixar_rastro=deixar_rastro)

    def detectObstacles(self, frame):
        return self.navModel.detectObstacles(frame)

    def analyzeTerrain(self, frame):
        return self.navModel.analyzeTerrain(frame)

    def getTrackHistory(self):
        return self.assetModel.get_tracks()


# --------------------------------------------------------
# LOOP PRINCIPAL – REPRODUÇÃO EXATA DO SEU SCRIPT ORIGINAL
# --------------------------------------------------------
def main():
    print("Iniciando captura de vídeo...")

    # Tenta abrir a câmera com MSMF, senão cai no fallback
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    except:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Câmera conectada.")

    # Cria modelos
    asset_model = AssetDetectionModel(MODEL_PATH)
    nav_model = NavigationModel()
    vision = VisionSystem(asset_model, nav_model)

    seguir = False
    deixar_rastro = False

    win = "YOLOv8 - Webcam (OO Version)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        success, img = cap.read()
        if not success:
            print("Falha ao capturar imagem.")
            break

        # Chamada ao VisionSystem → AssetDetectionModel → YOLO
        img_plot, results = vision.detectAssets(img, seguir=seguir, deixar_rastro=deixar_rastro)

        # Evita janela preta caso falhe o plot
        display = img_plot if img_plot is not None else img

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("Encerrando...")
            break

        if key == ord('s'):
            seguir = not seguir
            print("seguir =", seguir)

        if key == ord('r'):
            deixar_rastro = not deixar_rastro
            print("deixar_rastro =", deixar_rastro)

        # Fecha automaticamente se a janela for fechada manualmente
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                print("Janela fechada.")
                break
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------
# EXECUÇÃO DIRETA SEM ARGUMENTOS
# --------------------------------------------------------
if __name__ == "__main__":
    main()
