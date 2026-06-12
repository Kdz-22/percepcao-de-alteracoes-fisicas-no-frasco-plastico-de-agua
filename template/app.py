import cv2
import numpy as np
import tensorflow as tf
import time

# =========================
# CONFIGURAÇÕES
# =========================

MODEL_PATH = "modelo_garrafas.keras"

CLASSES = [
    "amassada",
    "garrafa_ok",
    "rotulo_torto",
    "sem_rotulo",
    "sem_tampa",
    "tampa_errada"
]

IMG_SIZE = (224, 224)
THRESHOLD = 0.50

# =========================
# CARREGA MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -------------------------
    # Pré-processamento
    # -------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # -------------------------
    # Predição
    # -------------------------

    preds = model.predict(img, verbose=0)[0]

    resultados = {
        classe: float(prob)
        for classe, prob in zip(CLASSES, preds)
    }

    # -------------------------
    # Lógica de negócio
    # -------------------------

    defeitos = []

    for classe, prob in resultados.items():

        if classe == "garrafa_ok":
            continue

        if prob >= THRESHOLD:
            defeitos.append((classe, prob))

    defeitos.sort(key=lambda x: x[1], reverse=True)

    garrafa_ok_prob = resultados["garrafa_ok"]

    if len(defeitos) == 0 and garrafa_ok_prob >= THRESHOLD:
        status = "APROVADA"
        status_color = (0, 255, 0)
    else:
        status = "REPROVADA"
        status_color = (0, 0, 255)

    # -------------------------
    # Painel lateral
    # -------------------------

    cv2.rectangle(frame, (0, 0), (420, 220), (30, 30, 30), -1)

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    y = 75

    if status == "APROVADA":

        cv2.putText(
            frame,
            f"Garrafa OK ({garrafa_ok_prob*100:.1f}%)",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Defeitos detectados:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y += 35

        for defeito, prob in defeitos[:4]:

            texto = f"- {defeito} ({prob*100:.1f}%)"

            cv2.putText(
                frame,
                texto,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            y += 30

    # -------------------------
    # FPS
    # -------------------------

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -------------------------
    # Exibe vídeo
    # -------------------------

    cv2.imshow("Projeto VISC - Inspecao Inteligente", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()