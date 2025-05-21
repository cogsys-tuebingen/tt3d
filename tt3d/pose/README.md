# TT Pose

How to get the 3D pose of both table tennis players from video

- mmpose MotionBert only gives the relative position
- Alphapose tracker does not work well and there are more performant 2d pose estimation models
- Sapiens:
    - 0.3b model is very heavy takes 0.5-1s/frame
    - has no tracking implemented
    - processes frames
-DWPose:
    - Faster but same problem as Sapiens


- MotionBert does not return perfectly flat feet
- Best 2D pose estimation
- Simple reprojection error does not give optimal resuts because of scaling and depth issue


## Pipeline
- Activate conda env 'openmmlab'
- Generate the 2D pose using mmpose with rtmpose (tracking activated)
```
 python generate_2d_pose.py  --input /data/online_tt_dataset/matches/25/rallies_videos/017.mp4  --output-root  vis_results   --save-predictions
```
- Correct the tracking
- Convert json (need to change the name in the script) TODO
```
python cvt_json.py
```
- Generate 3D pose with MotionBert (saved in repo)
```
python infer_wild.py --vid_path /home/gossard/Git/tt3d/data/demo_video/ma_lebrun_001.mp4 --json_path /home/gossard/Git/tt3d/vis_results/results_ma_lebrun_001_2d_mb.json --out_path /home/gossard/Git/tt3d/vis_results/ma_lebrun_001_mb --focus 0

```
- Transform the pose from camera coordinates to world coordinates
```
python align.py 
```
