# PreciseCam: Precise Camera Control for Text-to-Image Generation

***[Edurne Bernal-Berdun](https://edurnebernal.github.io/)<sup>1</sup>, [Ana Serrano](https://ana-serrano.github.io/)<sup>1</sup>, [Belen Masia](https://webdiis.unizar.es/~bmasia/)<sup>1</sup>, [Matheus Gadelha](https://research.adobe.com/person/matheus-gadelha/)<sup>2</sup>, [Yannick Hold-Geoffroy](https://research.adobe.com/person/yannick-hold-geoffroy/)<sup>2</sup>, [Xin Sun](https://www.sunxin.name/)<sup>2</sup>, [Diego Gutierrez](http://giga.cps.unizar.es/~diegog/)<sup>1</sup>***

*<sup>1</sup>Universidad de Zaragoza - I3A, <sup>2</sup> Adobe Research*

📅 *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025*

## Abstract

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
**[Diffusers](https://huggingface.co/docs/diffusers/index)** library has been customed to support PreciseCam. Adapted fork:: [`edurnebernal/diffusers-adapted`](https://github.com/edurnebernal/diffusers-adapted) 


### 🧪 Demo
This project provides a Gradio-based demo for **PreciseCam**. Our PreciseCam model is trained to control Stable Diffusion XL. To run the demo:

```
python demo.py
```

Once the Gradio interface launches in your browser:

* **Set the Camera Parameters:** Use the sliders on the left to configure the Roll, Pitch, Vertical Field of View (FOV), and ξ (distortion parameter).

* **Generate the Perspective Field:** Click the "Compute PF-US" button. This will generate and display the PF-US image based on the camera parameters.

* **Enter a Prompt:** Write a custom prompt in the textbox (e.g., "A colorful autumn park with leaves of orange, red, and yellow scattered across a winding path").

* **Generate the Final Image:** Click the "Generate Image" button. The system will use the perspective fields and the prompt to synthesize a final image.


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



