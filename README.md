# PreciseCam: Precise Camera Control for Text-to-Image Generation

***[Edurne Bernal-Berdun](https://edurnebernal.github.io/)<sup>1</sup>, [Ana Serrano](https://ana-serrano.github.io/)<sup>1</sup>, [Belen Masia](https://webdiis.unizar.es/~bmasia/)<sup>1</sup>, [Matheus Gadelha](https://research.adobe.com/person/matheus-gadelha/)<sup>2</sup>, [Yannick Hold-Geoffroy](https://research.adobe.com/person/yannick-hold-geoffroy/)<sup>2</sup>, [Xin Sun](https://www.sunxin.name/)<sup>2</sup>, [Diego Gutierrez](http://giga.cps.unizar.es/~diegog/)<sup>1</sup>***

*<sup>1</sup>Universidad de Zaragoza - I3A, <sup>2</sup> Adobe Research*

📅 *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025*

## 📝 Abstract

Images as an artistic medium often rely on specific camera angles and lens distortions to convey ideas or emotions; however, such precise control is missing in current text-to-image models. We propose an efficient and general solution that allows precise control over the camera when generating both photographic and artistic images. Unlike prior methods that rely on predefined shots, we rely solely on four simple extrinsic and intrinsic camera parameters, removing the need for pre-existing geometry, reference 3D objects, and multi-view data. We also present a novel dataset with more than 57,000 images, along with their text prompts and ground-truth camera parameters. Our evaluation shows precise camera control in text-to-image generation, surpassing traditional prompt engineering approaches.

🔗 [**📄 Paper on arXiv**](https://arxiv.org/abs/2501.12910) | [**🌐 Project Page**](https://graphics.unizar.es/projects/PreciseCam2024/)


---

### 📦 Model Access

The model is available on Hugging Face: [`edurnebb/PreciseCam`](https://huggingface.co/edurnebb/PreciseCam)

***NOTE:*** *We offer a public model that differs from the one used in the paper. While results may vary, the overall behavior remains consistent.*



---

### ⚙️ Installation

To set up the environment with Conda and install dependencies, simply run:

```bash
conda create -n precisecam --yes
conda activate precisecam

bash environment_setup.sh
```

This project uses a custom fork of the 🤗 **[Diffusers](https://huggingface.co/docs/diffusers/index)** library to enable camera control support:

🔧 Forked version: [`edurnebernal/diffusers-adapted`](https://github.com/edurnebernal/diffusers-adapted) 

---

### 🧪Running the Demo
We provide a Gradio-based demo for **PreciseCam**. Our PreciseCam model is trained to control Stable Diffusion XL. 

To launch the demo:

```
python demo.py
```

Once the Gradio interface launches in your browser:

* **Set the Camera Parameters:** Use the sliders on the left to configure the Roll, Pitch, Vertical Field of View (FOV), and ξ (distortion parameter).

* **Generate the Perspective Field:** Click the "Compute PF-US" button. This will generate and display the PF-US image based on the camera parameters.

* **Enter a Prompt:** Write a custom prompt in the textbox (e.g., "A colorful autumn park with leaves of orange, red, and yellow scattered across a winding path").

* **Generate the Final Image:** Click the "Generate Image" button. The system will use the perspective fields and the prompt to synthesize a final image.

*The demo has been tested on a NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)*

---

### 🖼️ Dataset Generation

To reproduce our dataset, download 360º panoramas from the following publicly available sources:

| Dataset                                       | Description / Paper                                                                 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [360-SOD](http://cvteam.net/projects/JSTSP20_DDS/DDS.html)            | Jia Li, Jinming Su, Changqun Xia, and Yonghong Tian. Distortion-adaptive salient object detection in 360º omnidirectional images. IEEE Journal of Selected Topics in Signal Processing, 2019.               |
| [CVRG-Pano](https://github.com/semihorhan/semseg-outdoor-pano)                | Semih Orhan and Yalin Bastanlar. Semantic segmentation of outdoor panoramic images. Signal, Image and Video Processing, 2022.      |
| [F-360iSOD](https://github.com/YeeZ93/F-360iSOD)    | Yi Zhang, Lu Zhang, Wassim Hamidouche, and Olivier Deforges. A fixation-based 360 benchmark dataset for salient object detection. In ICIP, 2020.              |
| [Poly Haven HDRIs](https://polyhaven.com/hdris)          | Poly Haven - HDRIs dataset. |
| [Sitzmann et al.](https://www.vincentsitzmann.com/vr-saliency/) | Vincent Sitzmann, Ana Serrano, Amy Pavel, Maneesh Agrawala, Diego Gutierrez, Belen Masia, and Gordon Wetzstein. Saliency in VR: How do people explore virtual environments? IEEE TVCG, 2018.         |
| [360Cities](https://www.360cities.net/)                  | 360cities dataset (*licence required*).           |


To generate crops and prompts:

```
python script.py --dataset_path ./dataset/panoramas --output_dir ./dataset --obtain_prompt
```

This script processes the panoramic images to generate the RGB image crops along with their corresponding PF-US. Optionally, it can also generate the textual prompts describing the crops using the BLIP-2 model.

##### Script Options



| Argument          | Type   | Default               | Description                                                       |
| ----------------- | ------ | --------------------- | ----------------------------------------------------------------- |
| `--dataset_path`  | `str`  | `./dataset/panoramas` | Path to the folder containing panorama images (`.jpg` or `.png`). |
| `--output_dir`    | `str`  | `./dataset`           | Directory to save output crops, PF-US maps, and prompts.               |
| `--obtain_prompt` | `flag` | `False`               | If set, uses BLIP-2 to generate a text prompt for each image crop. |
| `--h`             | `int`  | `1024`                | Height of the output image crops.                                 |
| `--w`             | `int`  | `1024`                | Width of the output image crops.                                  |

If `--obtain_prompt` is enabled, the script will load the [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) model. Make sure your system has enough GPU memory (ideally ≥8GB) to load the model. The script will automatically use GPU if available.

##### Output Structure
```
output_dir/
├── images/          # Image crops from panoramas
├── pf_us/           # Corresponding PF-US maps
└── prompts.jsonl    # (Optional) BLIP-2 prompts with image and PF-US paths
```
---

### 📖 Citations

**PreciseCam:**

```bibtex
@article{bernal2025precisecam,
  title={PreciseCam: Precise Camera Control for Text-to-Image Generation},
  author={Bernal-Berdun, Edurne and Serrano, Ana and Masia, Belen and Gadelha, Matheus and Hold-Geoffroy, Yannick and Sun, Xin and Gutierrez, Diego},
  journal={arXiv preprint arXiv:2501.12910},
  year={2025}
}
```

**Diffusers Library:**

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

**BLIP-2:**
```bibtex
@inproceedings{li2023blip2,
      title={{BLIP-2:} Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models}, 
      author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
      year={2023},
      booktitle={ICML},
}
```



