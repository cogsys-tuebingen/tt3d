# 3D Reconstruction Benchmark

We recorded with a multi-camera setup the 3D ball trajectory of 2 amateur table tennis players. From them we generated synthetic 2D ball positioin observations. This was done to benchmark the 3D ball trajectory reconstruction. The synthetic observations were generated for 3 different points of view (side, oblique and back) common in table tennis match recordings. The corresponding camera parameters are saved in the yaml files. We also generated the observations with noise to asses the robustess of the algorithm to inaccurate ball detections.

The frame used here is X along side width of the table, Y along the length of  the table and Z upwards. The origin of the frame is set at the center of the table's surface.

Each cvs file contains the timestamp in s, the ground truth 3D position in meters and the the camera coordinates of the ball (u, v). We additionnaly simulated the blur information for the ball with (l, theta).
