# Player Re-Identification in Sports Footage

## Overview
This project implements a solution for player re-identification in sports videos, ensuring that the same player retains the same ID across different camera feeds (broadcast and tacticam). It uses a YOLOv11-based object detector and simple appearance-based matching for cross-camera player mapping.

## Features
- Detects players in both broadcast and tacticam videos using a provided YOLOv11 model.
- Extracts appearance features (color histograms) for each detected player.
- Matches players across videos using feature similarity and spatial heuristics.
- Visualizes results by drawing consistent IDs on both videos.

## Project Structure
```
player_reid/
├── README.md
├── requirements.txt
├── config.yaml
├── detect.py
├── reid.py
├── pr_utils.py
├── main.py
├── data/
│   ├── broadcast.mp4
│   └── tacticam.mp4
└── weights/
    └── yolo.pt
```

## Setup
1. Clone this repository or download the code.
2. Download the YOLOv11 weights from the provided link and place them in `weights/yolo.pt`.
3. Place the two video files (`broadcast.mp4` and `tacticam.mp4`) in the `data/` directory.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main pipeline:
```bash
python main.py
```

- The output will be annotated videos in the `output/` directory and a mapping file showing player ID correspondences.

## Notes
- The detection model is a fine-tuned YOLOv11 for player and ball detection.
- The matching is based on color histograms and bounding box overlap for simplicity.
- Visualization is provided but kept simple for clarity.
