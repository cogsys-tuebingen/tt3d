from calibrate import calibrate_frame
from pathlib import Path
from table_calibrator import TableCalibrator


if __name__ == "__main__":
    dir_path = Path("/home/gossard/Git/tt3d/data/calibration_test/imgs/")
    img_paths = sorted((dir_path / "original").glob("[0-9][0-9][0-9].png"))
    # img_paths = [str(x) for x in img_paths]
    calibrator = TableCalibrator()
    for path in img_paths[:]:
        rvec, tvec, f = calibrate_frame(
            path,
            calibrator=calibrator,
            output_path=dir_path / "calibrated" / f"{path.stem}_calibrated.png",
        )
