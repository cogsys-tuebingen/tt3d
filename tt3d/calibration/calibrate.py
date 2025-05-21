"""
Script to calibrate the camera from the video
"""

import argparse
from pathlib import Path
from table_segmenter import TableSegmenter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
from albumentations.pytorch import ToTensorV2
from table_calibrator import TableCalibrator
from utils import save_camcal


def calibrate_frame(image_path, calibrator=None, display=False, output_path=None):
    """
    Calibrates a single frame by extracting rotation and translation vectors.

    Args:
        image_path (str): Path to the input image.
        display (bool): Whether to display the debug image.
        output_path (str, optional): Path to save the debug image.

    Returns:
        tuple: (rvec, tvec, f) containing rotation vector, translation vector, and focal length.
    """
    # Load the image
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = frame.shape[:2]
    if calibrator is None:
        calibrator = TableCalibrator(height, width)
    calibrator.set_size(height, width)

    # Process the image
    rvec, tvec, f, er, debug_img = calibrator.process(frame, debug=True)

    # Display if required
    if display:
        cv2.imshow("Calibration", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the debug image if an output path is provided
    if output_path:
        cv2.imwrite(str(output_path), debug_img)
        print("Image saved:", output_path)

    return rvec, tvec, f


def calibrate_video(input_video_path, display=False, render=False, output_path=None):
    rvecs, tvecs, fs = [], [], []
    # Load the video
    cap = cv2.VideoCapture(str(input_video_path))

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    calibrator = TableCalibrator(height, width)

    # Define the codec and create VideoWriter object
    if render:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("Processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rvec, tvec, f, er, debug_img = calibrator.process(frame, debug=True)
        if rvec is None:
            rvecs.append(np.zeros((3, 1)))
            tvecs.append(np.zeros((3, 1)))
            fs.append(0)
        else:
            rvecs.append(rvec)
            tvecs.append(tvec)
            fs.append(f)

        if display:
            cv2.imshow("Calibration", debug_img)
            cv2.waitKey(1)

        if render:
            # Write the frame to the output video
            out.write(debug_img)

    # Release resources
    cap.release()
    if render:
        out.release()
        print("Video saved: ", output_path)
    return rvecs, tvecs, fs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("-o", "--output", help="Path for the csv")
    parser.add_argument(
        "--display",
        help="Display the generated segmentation frames",
        action="store_true",
    )
    parser.add_argument(
        "--render",
        help="Generates the video",
        action="store_true",
    )

    args = parser.parse_args()
    print("Video path: ", args.video_path)
    input_path = Path(args.video_path)
    video_output_path = input_path.parent / f"{input_path.stem}_calibrated.mp4"

    rvecs, tvecs, fs = calibrate_video(
        input_path,
        render=args.render,
        display=args.display,
        output_path=video_output_path,
    )

    if args.output:
        save_camcal(args.output, rvecs, tvecs, fs)
