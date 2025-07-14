import cv2
import yaml
from detect import PlayerDetector, extract_player_crops
from reid import match_players
from pr_utils import draw_boxes, ensure_dir
import os

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def process_first_frame(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, [], []
    detections = detector.detect(frame)
    print("Detections:", detections)
    crops = extract_player_crops(frame, detections)
    cap.release()
    return frame, detections, crops

def main():
    config = load_config()
    ensure_dir(config['output']['dir'])

    detector = PlayerDetector(config['weights']['yolo'])

    # Process only the first frame of both videos
    frame1, dets1, crops1 = process_first_frame(config['data']['broadcast_video'], detector)
    frame2, dets2, crops2 = process_first_frame(config['data']['tacticam_video'], detector)

    if frame1 is None or frame2 is None:
        print("Error: Could not read frames from one or both videos.")
        return
    print(f"Broadcast: {len(crops1)} players, Tacticam: {len(crops2)} players")
    # Use first frame for mapping (for simplicity)
    matches = match_players(crops1, crops2)

    # Assign IDs
    ids1 = list(range(len(dets1)))
    ids2 = [-1]*len(dets2)
    for i1, i2 in matches.items():
        ids2[i2] = ids1[i1]

    # Visualize and save first frame with IDs
    out1 = draw_boxes(frame1.copy(), dets1, ids1)
    out2 = draw_boxes(frame2.copy(), dets2, ids2, color=(255,0,0))
    cv2.imwrite(os.path.join(config['output']['dir'], 'broadcast_annotated.jpg'), out1)
    cv2.imwrite(os.path.join(config['output']['dir'], 'tacticam_annotated.jpg'), out2)

    # Save mapping
    with open(os.path.join(config['output']['dir'], 'player_mapping.txt'), 'w') as f:
        for i1, i2 in matches.items():
            f.write(f"Broadcast ID {i1} <-> Tacticam ID {i2}\n")

if __name__ == '__main__':
    main()