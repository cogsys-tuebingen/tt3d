"""
Script to convert 2d json format to one accepted by MotionBert
"""
import json
import argparse
import numpy as np
from pathlib import Path


def get_array(input):
    return np.array(input["__ndarray__"], dtype=input["dtype"])


def convert_json(input_path):
    input_path = Path(input_path)
    output_path = input_path.parent / "mb_input.json"

    with open(input_path, "r") as file:
        data = json.load(file)

    n_frames = len(data["instance_info"])
    output = []

    for frame_idx, frame_data in enumerate(data["instance_info"]):
        for h_data in frame_data["instances"]:
            track_id = h_data["attributes"]["track_id"]
            kps = get_array(
                h_data["attributes"]["_pred_instances"]["attributes"]["keypoints"]
            )[0]
            scores = get_array(
                h_data["attributes"]["_pred_instances"]["attributes"]["keypoint_scores"]
            ).T
            kps_and_score = np.hstack([kps, scores]).flatten().tolist()
            box = get_array(
                h_data["attributes"]["_pred_instances"]["attributes"]["bboxes"]
            )[0].tolist()
            box_score = get_array(
                h_data["attributes"]["_pred_instances"]["attributes"]["bbox_scores"]
            )[0]
            instance = {
                "image_id": f"{frame_idx}.jpg",
                "category_id": int(1),
                "keypoints": kps_and_score,
                # "score": box_score,
                "box": box,
                "idx": int(track_id),
            }
            print(instance)
            output.append(instance)
            # print(pose)
            # plot_skeleton_2d(pose[0])
            # plt.show()
            # print(data[0][0]["__instance_type__"])
            # print(data[0][0]["attributes"].keys())
            # print(data[0][0]["attributes"]["_pred_instances"].keys())
            # print(data[0][0]["attributes"]["_pred_instances"]["attributes"])
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 2D JSON format to MotionBert-compatible format."
    )
    parser.add_argument("input", type=str, help="Path to the input JSON file")
    args = parser.parse_args()

    convert_json(args.input)
