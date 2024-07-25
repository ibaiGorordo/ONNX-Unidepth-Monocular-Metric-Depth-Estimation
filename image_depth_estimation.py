import cv2
import numpy as np
from imread_from_url import imread_from_url

from unidepth import UniDepth, colorize_map

model_path = "models/unidepthv2_vits14_simp.onnx"

# Initialize UniDepth model
unidepth = UniDepth(model_path)

img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/ZZtopthuis.JPG/1280px-ZZtopthuis.JPG"
img = imread_from_url(img_url)

# Estimate depth
depth, confidence_map, camera_intrinsics = unidepth(img)

# Visualize depth map
depth_map = colorize_map(depth)
combined = np.hstack([img, depth_map])
cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.imshow("Depth Map", combined)

# Visualize confidence map
confidence_map = colorize_map(confidence_map, max_value=1, invert=False)
cv2.imshow("Confidence Map", confidence_map)
cv2.waitKey(0)

cv2.imwrite("doc/img/estimated_depth.png", combined)