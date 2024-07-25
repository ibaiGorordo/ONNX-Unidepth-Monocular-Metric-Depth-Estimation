import cv2
from cap_from_youtube import cap_from_youtube

from unidepth import UniDepth, colorize_map

model_path = "models/unidepthv2_vits14_simp.onnx"

# Initialize UniDepth model
unidepth = UniDepth(model_path)

# Initialize video
# cap = cv2.VideoCapture("input.mp4")
videoUrl = 'https://youtu.be/b8eHg1YSUKg?si=ZO5OVYng_e4FhidE'
cap = cap_from_youtube(videoUrl, resolution='1440p')
start_time = 60 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

max_depth = 10 # meters
cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
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

    # out.write(combined)

cap.release()
# out.release()
