# ONNX-UniDepth Monocular Metric Depth Estimation

![!ONNX-UniDepth Monocular Metric Depth Estimation](https://github.com/ibaiGorordo/ONNX-Unidepth-Monocular-Metric-Depth-Estimation/raw/main/doc/img/estimated_depth.png)


# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```shell
git clone https://github.com/ibaiGorordo/ONNX-Unidepth-Monocular-Metric-Depth-Estimation.git
cd ONNX-Unidepth-Monocular-Metric-Depth-Estimation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
- Download the model from [Hugging Face](https://huggingface.co/ibaiGorordo/unidepth-v2-vits14-onnx/blob/main/unidepthv2_vits14_simp.onnx) and save it in the **models** folder.
- Otherwise use this Google Colab notebook to convert the model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TV5WoVmBqubV4TZ_NKhy0KAuIgCPtZHU?usp=sharing)

# Original UniDepth model
The original UniDepth model can be found in this repository: [UniDepth Repository](https://github.com/lpiccinelli-eth/UniDepth)
- The License of the models is Attribution-NonCommercial 4.0 International: [License](https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE)

# Examples

 * **Image inference**:
 ```shell
 python image_depth_estimation.py
 ```

 * **Webcam inference**:
 ```shell
 python webcam_depth_estimation.py
 ```

 * **Video inference**:
 ```shell
 python video_depth_estimation.py
 ```

 * **Rerun SDK visualization**: https://youtu.be/NBkWIlEIEi0?si=YhBbaVaAlSyVIpit
 ```shell
 python video_depth_estimation.py
 ```

 ![!Unidepth Rerun SDK visualization](https://github.com/ibaiGorordo/ONNX-Unidepth-Monocular-Metric-Depth-Estimation/raw/main/doc/img/rerun_video.gif)

  *Original video: [https://www.youtube.com/watch?v=8jsXi_51B40](https://www.youtube.com/watch?v=8jsXi_51B40)*

# References:
* UniDepth model: [https://github.com/lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
* Rerun SDK: [https://github.com/rerun-io/rerun](https://github.com/rerun-io/rerun)