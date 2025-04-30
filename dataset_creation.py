import os
import cv2
import utils
import tqdm
import argparse
from PIL import Image


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop images from panorama and obtain perspective fields"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset/panoramas",
        help="Path to the folder containing the panorama images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset",
        help="Directory to save the generated image crops and PF-US",
    )
    parser.add_argument(
        "--obtain_prompt",
        action="store_true",
        default=False,
        help="If set, the script will obtain a prompt for the generated images using BLIP",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=1024,
        help="Height of the crop images",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=1024,
        help="Width of the crop images",
    )
    return parser.parse_args()

def BLIP_setup():
    # Setup the device and model for BLIP
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # Load the BLIP processor and model
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto"
    )
    model.to(device)
    
    return processor, model, device
    
def main():
    print("Starting dataset creation...")
    args = parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    obtain_prompt = args.obtain_prompt
    h = args.h
    w = args.w
    
    if obtain_prompt:
        import torch
        processor, model, device = BLIP_setup()
        print("BLIP model loaded successfully.")
        

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pf_us"), exist_ok=True)

    # Get panorama images
    panorama_images = [
        os.path.join(dataset_path, img)
        for img in os.listdir(dataset_path)
        if img.endswith(".jpg") or img.endswith(".png")
    ]

    panorama_images.sort()
    print(f"Found {len(panorama_images)} panorama images.")

    # Generate crops and PF-US for each panorama image
    for panorama_path in tqdm.tqdm(panorama_images, desc="Processing panoramas"):
        panorama = cv2.imread(panorama_path)

        # Sample camera parameters for the panorama
        sampled_parameters = utils.sample_parameters()

        for i, params in enumerate(sampled_parameters):
            roll, yaw, pitch, vfov, xi = params

            # Obtain image crop and PF-US
            image, pf_us = utils.obtain_pf(panorama, roll, yaw, pitch, vfov, xi, h, w)

            # Save PF-US and blend images
            image_filename = os.path.join(output_dir, "images", f"image_{i}.png")
            pf_us_filename = os.path.join(output_dir, "pf_us", f"pf_us_{i}.png")
            cv2.imwrite(image_filename, image)
            cv2.imwrite(pf_us_filename, pf_us)
            
            if obtain_prompt:
                # Obtain prompt using BLIP
                image_pil = Image.fromarray(image).convert('RGB')
                inputs = processor(image_pil, return_tensors="pt").to(device, torch.float16)
                outputs = model.generate(**inputs)
                prompt = processor.decode(outputs[0], skip_special_tokens=True).replace("\n", "")
                print(f"Generated prompt: {prompt}")
                
                # Save the prompt in the jsonl file
                with open(os.path.join(output_dir, "prompts.jsonl"), "a") as f:
                    f.write(f"text: {prompt}, image: {image_filename}, 'conditioning_image': {pf_us_filename}\n")


if __name__ == "__main__":
    main()
