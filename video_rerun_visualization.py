import cv2
from cap_from_youtube import cap_from_youtube
import rerun as rr

from unidepth import UniDepth, colorize_map

model_path = "models/unidepthv2_vits14_simp.onnx"

# Initialize UniDepth model
unidepth = UniDepth(model_path)

# Initialize video
# cap = cv2.VideoCapture("input.mp4")
videoUrl = 'https://youtu.be/8jsXi_51B40?si=Iq8WfiJdvOSac-p-'
cap = cap_from_youtube(videoUrl, resolution='1440p60')
start_time = 15 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

max_depth = 10 # meters
rr.init("Unidepth")
rr.connect()
rr.spawn()
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Estimate depth
    depth, confidence_map, camera_intrinsics = unidepth(frame)
    depth = depth.clip(0, max_depth)


    rr.log("world/camera/image",
           rr.Pinhole(
                resolution=[frame.shape[1], frame.shape[0]],
                focal_length=0.8 * frame.shape[1]
           ),
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rr.log("world/camera/image/rgb", rr.Image(rgb).compress(jpeg_quality=95))

    rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1))

cap.release()