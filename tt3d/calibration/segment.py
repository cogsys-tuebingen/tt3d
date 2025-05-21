"""
Script to run the segmentation on a video
"""

import argparse
from pathlib import Path
from table_segmenter import TableSegmenter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from table_segmenter import TableSegmenter
from torchvision import transforms as T
from utils import postprocess, preprocess


def overlay_segmentation_mask(img, mask, color=(0, 150, 150), alpha=0.8):
    """
    Overlay a segmentation mask on an image.

    Args:
        img (np.ndarray): The input image (H, W, 3).
        mask (np.ndarray): The segmentation mask (H, W), with values in [0, 255].
        color (tuple): RGB color for the mask overlay.
        alpha (float): Transparency factor (0: no overlay, 1: full mask).

    Returns:
        np.ndarray: Image with overlayed segmentation mask.
    """
    img = np.array(img, dtype=np.uint8)

    # Ensure mask is binary (0 or 255)
    mask = (mask > 127).astype(np.uint8) * 255

    # Convert mask to 3-channel
    mask_colored = np.zeros_like(img, dtype=np.uint8)
    for i in range(3):
        mask_colored[..., i] = mask * (color[i] / 255.0)

    # Blend the mask with the image
    overlay = cv2.addWeighted(img, 1.0, mask_colored, alpha, 0)

    return overlay


def segment_video(
    input_video_path, model, display=False, render=False, output_path=None
):
    output_masks = []
    # Load the video
    cap = cv2.VideoCapture(str(input_video_path))

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    if render:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Ensure model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the model input size
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sizes = [pil_img.size[::-1]]

        # Run the table segmentation model
        with torch.no_grad():
            t_imgs = preprocess([pil_img])
            t_masks = segmenter.infer(t_imgs)
            masks = postprocess(t_masks, sizes)
        output_masks.append(masks[0])

        if display:
            display_frame = cv2.hconcat(
                [frame, cv2.cvtColor(masks[0], cv2.COLOR_GRAY2RGB).astype(np.uint8)]
            )
            display_frame = cv2.resize(display_frame, (1000, 300))
            cv2.imshow("Segmentation", display_frame)
            cv2.waitKey(1)

        output_frame = overlay_segmentation_mask(frame, masks[0])

        if render:
            # Write the frame to the output video
            out.write(output_frame)

    # Release resources
    cap.release()
    if render:
        out.release()
        print("Video saved: ", output_path)
    return np.array(output_masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("-o", "--output", help="Path for the output video")
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
    if args.render:
        if not args.output:
            output_path = input_path.parent / f"{input_path.name}_segmented.mp4"
        else:
            output_path = Path(args.output)
    else:
        output_path = None

    model_path = Path("./weights/table_segmentation.ckpt")
    # Load the model
    segmenter = TableSegmenter.load_from_checkpoint(model_path)

    # Process video
    masks = segment_video(
        args.video_path,
        segmenter,
        display=args.display,
        render=args.render,
        output_path=output_path,
    )
