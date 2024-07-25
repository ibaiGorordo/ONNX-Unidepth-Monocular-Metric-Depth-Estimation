import time
import cv2
import numpy as np
import onnxruntime


def colorize_map(map, max_value=-1, invert=True):
    if max_value < 0:
        max_value = np.max(map)

    map = np.clip(map, 0, max_value)
    map = (map / max_value * 255).astype(np.uint8)
    if invert:
        map = 255 - map
    map = cv2.applyColorMap(map, cv2.COLORMAP_MAGMA)
    return map


class UniDepth:
    def __init__(self, path):
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image):
        return self.estimate_depth(image)

    def estimate_depth(self, image):
        input_tensor = self.prepare_input(image)

        outputs = self.inference(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, outputs):
        camera_intrinsics = np.squeeze(outputs[0])
        depth = np.squeeze(outputs[1])
        confidence_map = np.squeeze(outputs[2])

        depth = cv2.resize(depth, (self.img_width, self.img_height))
        confidence_map = cv2.resize(confidence_map, (self.img_width, self.img_height))

        return depth, confidence_map, camera_intrinsics

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/unidepthv2_vits14_simp.onnx"

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
