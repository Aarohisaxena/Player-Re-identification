from ultralytics import YOLO

# Set the class index for "player" as per your model's mapping.
PLAYER_CLASS_INDEX = 2  # Change this if your model uses a different index for players.
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed.

class PlayerDetector:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        results = self.model(frame)
        players = []
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                # Debug: print all detected classes and confidence
                print(f"Detected class: {cls_idx}, confidence: {conf:.2f}")
                # Only keep detections for the whole player with sufficient confidence
                if cls_idx == PLAYER_CLASS_INDEX and conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    players.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
        print("Players extracted:", players)  # Debug print
        return players

def extract_player_crops(frame, detections):
    crops = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
    return crops