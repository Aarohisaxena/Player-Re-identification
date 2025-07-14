import cv2
import numpy as np

def extract_color_histogram(image, bins=(8, 8, 8)):
    if image is None or image.size == 0:
        return np.zeros(bins[0]*bins[1]*bins[2])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def match_players(crops1, crops2, threshold=1.0):
    features1 = [extract_color_histogram(crop) for crop in crops1]
    features2 = [extract_color_histogram(crop) for crop in crops2]
    matches = {}
    for i, feat1 in enumerate(features1):
        best_j, best_score = -1, float('inf')
        for j, feat2 in enumerate(features2):
            score = cv2.compareHist(feat1, feat2, cv2.HISTCMP_BHATTACHARYYA)
            print(f"Player {i} (broadcast) vs Player {j} (tacticam): score={score:.3f}")
            if score < best_score:
                best_score = score
                best_j = j
        if best_score < threshold:
            matches[i] = best_j
    return matches