import cv2
import argparse
import time
import logging
from ultralytics import YOLO
import os

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# Argument Parser
# -------------------------------
parser = argparse.ArgumentParser(description="Conveyor Belt Defect Detection using YOLOv8")
parser.add_argument("--image", type=str, help="Path to input image")
parser.add_argument("--video", type=str, help="Path to input video")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default=0.5)")
args = parser.parse_args()

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "best.pt")
model_path = os.path.abspath(model_path)

if not os.path.exists(model_path):
    logging.error(f"Model file not found at: {model_path}")
    exit()

model = YOLO(model_path)
logging.info("Model loaded successfully.")

# -------------------------------
# IMAGE MODE
# -------------------------------
if args.image:
    if not os.path.exists(args.image):
        logging.error("Image file not found.")
        exit()

    frame = cv2.imread(args.image)
    results = model(frame, conf=args.conf)

    annotated_frame = results[0].plot()

    output_path = "../outputs/result_image.jpg"
    cv2.imwrite(output_path, annotated_frame)

    logging.info(f"Detection complete. Output saved at {output_path}")

    cv2.imshow("Image Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# VIDEO MODE
# -------------------------------
elif args.video:
    if not os.path.exists(args.video):
        logging.error("Video file not found.")
        exit()

    cap = cv2.VideoCapture(args.video)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf)
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Video Defect Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# WEBCAM MODE (Default)
# -------------------------------
else:
    logging.info("Starting webcam detection. Press 'q' to exit.")
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=args.conf)
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Webcam Defect Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
