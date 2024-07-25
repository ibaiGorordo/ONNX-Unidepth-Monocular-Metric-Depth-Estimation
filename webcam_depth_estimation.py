import cv2

from unidepth import UniDepth, colorize_map

model_path = "models/unidepthv2_vits14_simp.onnx"

# Initialize UniDepth model
unidepth = UniDepth(model_path)

cap = cv2.VideoCapture(0)

max_depth = 3 # meters
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Estimate depth
    depth, confidence_map, camera_intrinsics = unidepth(frame)

    # Visualize depth map
    depth_map = colorize_map(depth, max_depth)
    combined = cv2.addWeighted(frame, 0.5, depth_map, 0.7, 0)
    cv2.imshow("Depth Map", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
