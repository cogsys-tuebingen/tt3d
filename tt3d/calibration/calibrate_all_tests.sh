#!/bin/bash

# Base directory for the test videos
DATA_DIR="/home/gossard/Git/tt3d/data/calibration_test/videos"

# Loop through test video numbers from 00 to 11
for i in $(seq -w 00 11); do
    VIDEO_PATH="$DATA_DIR/test_${i}.mp4"
    
    if [[ -f "$VIDEO_PATH" ]]; then
        echo "Processing: $VIDEO_PATH"
        python src/calibration/calibrate.py "$VIDEO_PATH" --render 
    else
        echo "Skipping: $VIDEO_PATH (file not found)"
    fi
done
