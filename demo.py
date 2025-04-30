import gradio as gr
import numpy as np
import torch
from PIL import Image
from perspective_fields import pano_utils as pu
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
)

# Load models
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
controlnet = ControlNetModel.from_pretrained("edurnebb/PreciseCam", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    vae=vae,
)

pipe.enable_model_cpu_offload()
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


def inference(prompt, pf_image, n_steps=20, seed=13):
    """Generates an image based on the given prompt and perspective field image."""
    pf = Image.fromarray(
        np.concatenate(
            [np.expand_dims(pf_image[:, :, i], axis=-1) for i in [2, 1, 0]], axis=2
        )
    )
    pf_condition = pf.resize((1024, 1024))
    generator = torch.manual_seed(seed)

    return pipe(
        prompt,
        num_inference_steps=n_steps,
        generator=generator,
        image=pf_condition,
        controlnet_conditioning_scale=1.0,
        middle_res_only=True,
    ).images[0]


def obtain_pf(xi, roll, pitch, vfov):
    """Computes perspective fields given camera parameters."""
    w, h = (1024, 1024)
    equi_img = np.zeros((h, w, 3), dtype=np.uint8)
    x = -np.sin(np.radians(vfov / 2))
    z = np.sqrt(1 - x**2)
    f_px_effective = -0.5 * (w / 2) * (xi + z) / x

    crop, _, _, _, up, lat, _ = pu.crop_distortion(
        equi_img, f=f_px_effective, xi=xi, H=h, W=w, az=10, el=-pitch, roll=roll
    )

    gravity = (up + 1) * 127.5
    latitude = np.expand_dims((np.degrees(lat) + 90) * (255 / 180), axis=-1)
    pf_image = np.concatenate([gravity, latitude], axis=2).astype(np.uint8)
    blend = pu.draw_perspective_fields(crop, up, np.radians(np.degrees(lat)))

    return pf_image, blend


# Gradio UI
demo = gr.Blocks(theme=gr.themes.Soft())

with demo:
    gr.Markdown("""---""")
    gr.Markdown("""# PreciseCam: Precise Camera Control for Text-to-Image Generation""")
    gr.Markdown("""1. Set the camera parameters (Roll, Pitch, Vertical FOV, ξ)""")
    gr.Markdown("""2. Click "Compute PF-US" to generate the perspective field image""")
    gr.Markdown("""3. Enter a prompt for the image generation""")
    gr.Markdown("""4. Click "Generate Image" to create the final image""")
    gr.Markdown("""---""")

    with gr.Row():
        with gr.Column():
            roll = gr.Slider(-90, 90, 0, label="Roll")
            pitch = gr.Slider(-90, 90, 1, label="Pitch")
            vfov = gr.Slider(15, 140, 50, label="Vertical FOV")
            xi = gr.Slider(0.0, 1, 0.2, label="ξ")
            prompt = gr.Textbox(
                lines=4,
                label="Prompt",
                show_copy_button=True,
                value="A colorful autumn park with leaves of orange, red, and yellow scattered across a winding path.",
            )
            pf_btn = gr.Button("Compute PF-US", variant="primary")
        with gr.Row():
            pf_img = gr.Image(height=1024 // 4, width=1024 // 4, label="PF-US")
            condition_img = gr.Image(
                height=1024 // 4, width=1024 // 4, label="Internal PF-US (RGB)"
            )

        with gr.Column(scale=2):
            result_img = gr.Image(label="Generated Image", height=1024 // 2)
            inf_btn = gr.Button("Generate Image", variant="primary")
    gr.Markdown("""---""")

    pf_btn.click(
        obtain_pf, inputs=[xi, roll, pitch, vfov], outputs=[condition_img, pf_img]
    )
    inf_btn.click(inference, inputs=[prompt, condition_img], outputs=[result_img])

demo.launch(share=True)
