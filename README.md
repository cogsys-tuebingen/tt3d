# TT3D: Table tennis match 3D reconstruction

This is the code for TT3D, a table tennis 3D reconstruction method that works with monocular videos. It performs the camera calibration and ball trajectory 3D reconstruction. 

- [Project Page](https://cogsys-tuebingen.github.io/tt3d/)
- [Paper](https://arxiv.org/pdf/2504.10035)

![Render](./render_landscape.png)

##  Setup

## Camera Calibration
To test out the segmentation model:
```
python tt3d/calibration/segment.py input.mp4
```

You can process the whole video with the following command. It will output a csv file with the raw observations (no filtering).

To test out the calibration pipeline:
```
python tt3d/calibration/calibrate.py input.mp4 -o cam_cal.csv
```

Once the measurements have been generated, they need to filtered with a kalman filter that will also perform outlier rejection.
```
python tt3d/calibration/filter.py input.csv --so -w 1280 -he 720 --display

```


## Pose estimation
- Activate conda env 'openmmlab'
- Generate the 2D pose using mmpose with rtmpose (tracking activated)
```
 python process_video.py  --input /data/online_tt_dataset/matches/25/rallies_videos/017.mp4  --output-root  vis_results   --save-predictions
```
- Correct the tracking
- Convert json (need to change the name in the script) TODO
```
python cvt_json.py input.json
```
- Generate 3D pose with MotionBert
```
python infer_wild.py \
--vid_path <your_video.mp4> \
--json_path <alphapose-results.json> \
--out_path <output_path>
--focus 0
```
- Fuse pose
```

```

- Generate aligned pose
```
python tt3d/pose/align.py
```


## Ball trajectory reconstruction
- Extract the unique frames
```
python generate_unique_frames.py
```
- Run blurball and get the 2D ball position
```
python3 src/main.py --config-name=eval_blurball detector.model_path=/home/gossard/Git/blurball_train/outputs/main/blurball/best_model
```
- Run reconstruction
```
python tt3d/rally/rally.py root_dir
```
- Render the whole trajectory
```
python tt3d/pose/render.py
```

## Citing
If you find this work useful, please consider citing:
```bibtex
@InProceedings{gossard2025,
          author    = {Gossard, Thomas and Ziegler, Andreas and Zell, Andreas},
          title     = {TT3D: Table Tennis 3D Reconstruction},
          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
          month     = {June},
          year      = {2025}
          }
```

