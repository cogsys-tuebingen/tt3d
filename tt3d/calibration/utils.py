import albumentations as A
import csv
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
from torchvision import transforms as T
from torchvision.io import read_image
import yaml


def tensor2array(tensor):
    return tensor.detach().cpu().numpy()


def get_K(f, h, w):
    return np.array(
        [
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1],
        ]
    )


def write_camera_info(yaml_path, rvec, tvec, f, h, w):
    """
    Writes camera calibration information to a YAML file.

    Parameters:
        yaml_path (str): Path to the YAML file.
        rvec (np.ndarray): Rotation vector.
        tvec (np.ndarray): Translation vector.
        f (float): Focal length.
        h (int): Image height.
        w (int): Image width.
    """
    # Prepare the data to be saved
    camera_info = {
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "f": int(f),
        "h": int(h),
        "w": int(w),
    }

    # Write to the YAML file
    with open(yaml_path, "w") as file:
        yaml.safe_dump(camera_info, file)


def read_camera_info(yaml_path):
    """
    Reads camera calibration information from a YAML file.

    Parameters:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Camera calibration data including rotation vector (rvec),
              translation vector (tvec), and focal length (f).
    """
    # Load the YAML file
    with open(yaml_path, "r") as file:
        camera_info = yaml.safe_load(file)

    # Extract the relevant information
    rvec = camera_info.get("rvec")
    tvec = camera_info.get("tvec")
    w = camera_info.get("w")
    h = camera_info.get("h")
    f = camera_info.get("f")

    # Return the values as a dictionary
    return np.array(rvec), np.array(tvec), f, h, w


def read_camcal(csv_path):
    """
    Reads camera calibration data from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        rvecs (list of np.ndarray): List of rotation vectors.
        tvecs (list of np.ndarray): List of translation vectors.
        fs (list): Focal length values.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    # print(df)
    filtered_data = df[
        ~(
            (
                df[["f", "rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"]]
                == 0
            ).all(axis=1)
        )
    ]

    idxs = filtered_data["Index"]
    # Extract rotation vectors, translation vectors, and focal lengths
    rvecs = [
        np.array([row["rvec_x"], row["rvec_y"], row["rvec_z"]]).reshape(3, 1)
        for _, row in filtered_data.iterrows()
    ]
    tvecs = [
        np.array([row["tvec_x"], row["tvec_y"], row["tvec_z"]]).reshape(3, 1)
        for _, row in filtered_data.iterrows()
    ]
    fs = filtered_data["f"].tolist()  # Ensure fs is a list of focal lengths

    return (
        np.array(idxs),
        np.array(rvecs).reshape((-1, 3)),
        np.array(tvecs).reshape((-1, 3)),
        np.array(fs),
    )


def save_camcal(csv_path, rvecs, tvecs, fs, errors=None):
    """
    Saves camera calibration data into a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
        rvecs (list of np.ndarray): List of rotation vectors.
        tvecs (list of np.ndarray): List of translation vectors.
        fs (list): Focal length values.
    """
    # Prepare the data for DataFrame
    data = {
        "Index": [],
        "rvec_x": [],
        "rvec_y": [],
        "rvec_z": [],
        "tvec_x": [],
        "tvec_y": [],
        "tvec_z": [],
        "f": [],
        "error": [],
    }

    if errors is not None:
        # Populate the data dictionary
        for i, (rvec, tvec, f, er) in enumerate(zip(rvecs, tvecs, fs, errors)):
            data["Index"].append(i)
            data["rvec_x"].append(rvec[0])
            data["rvec_y"].append(rvec[1])
            data["rvec_z"].append(rvec[2])
            data["tvec_x"].append(tvec[0])
            data["tvec_y"].append(tvec[1])
            data["tvec_z"].append(tvec[2])
            data["f"].append(f)
            data["error"].append(er)
    else:
        for i, (rvec, tvec, f) in enumerate(zip(rvecs, tvecs, fs)):
            data["Index"].append(i)
            data["rvec_x"].append(rvec[0])
            data["rvec_y"].append(rvec[1])
            data["rvec_z"].append(rvec[2])
            data["tvec_x"].append(tvec[0])
            data["tvec_y"].append(tvec[1])
            data["tvec_z"].append(tvec[2])
            data["f"].append(f)

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


def PIL2tensor(pil_image):
    np_image = np.array(pil_image)
    augmented = ToTensorV2(transpose_mask=False)(image=np_image)
    return augmented["image"]


# Converts the image from PIL to tensor of the right size
# Takes as input a list of pil_images
def preprocess(pil_imgs, size=(640, 320), device="cuda:0"):
    transform = A.Compose([A.Resize(height=size[1], width=size[0]), ToTensorV2()])
    resized_tensors = []
    for img in pil_imgs:
        img_array = np.array(img)
        transformed = transform(image=img_array)
        resized_tensors.append(transformed["image"][None].to(device))
    return torch.cat(resized_tensors)


def postprocess(t_masks, sizes):
    num_masks = t_masks.shape[0]
    resized_masks = []

    for i in range(num_masks):
        mask = tensor2array(t_masks[i].squeeze())
        mask = 255 * mask
        resized_mask = cv2.resize(
            mask, (sizes[i][1], sizes[i][0]), interpolation=cv2.INTER_NEAREST
        )
        # print(resized_mask.shape)
        resized_masks.append(resized_mask)

    return resized_masks


def read_img(img_path):
    return Image.open(img_path).convert("RGB")
