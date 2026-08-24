#!/usr/bin/env python3

import requests
import json
import os
import argparse
import subprocess
from datetime import datetime
from tqdm import tqdm
from time import time
from io import BytesIO
from PIL import Image
import re

HOST = 'http://127.0.0.1:8888'

# PARAMETRI GLOBALI PER FOOOCUS
GUIDANCE_SCALE = 4.0 
INPAINT_STRENGTH = 0.9
IMAGE_SEED = 2668419004769029052


def inpaint_outpaint(params: dict, input_image: bytes, input_mask: bytes = None) -> dict:
    response = requests.post(
        url=f"{HOST}/v1/generation/image-inpaint-outpaint",
        data=params,
        files={"input_image": input_image,
               "input_mask": input_mask})
    return response.json()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_paths', type=str, required=True, help='Txt file containing the path of source images to be inpainted.')
    parser.add_argument('--masks_path', type=str, required=True, help='Directory containing the masks of the images.')
    parser.add_argument('--prompt', type=str, required=True, help='Directory containing the prompts generated for each image.')
    parser.add_argument('--save_path', type=str, required=True, help="Directory to store the final inpainted images.")
    parser.add_argument('--fooocus_api_dir', type=str, help='Location of Fooocus-API directory.')
    return parser 

def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    clean_prompts_path = os.path.join(args.save_path, "clean_prompts.jsonl")

    images = [f for f in os.listdir(args.prompt) if f.endswith('.json')]

    start = time()
    
    for img in tqdm(images, desc="Inpainting faces", bar_format=f'\033[36m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):

        # Estrazione pulita del nome immagine e del suffisso del volto (es. face0, face1)
        image_name = img[:img.rfind("_")]
        suffix = img[img.rfind("_")+1:img.find(".json")]

        with open(args.images_paths, "r") as f:
            paths = f.readlines()

        images_names = [path.split("/")[-1].replace(".jpg", "").replace(".png", "").strip() for path in paths]

        with open(os.path.join(args.prompt, img), 'r') as file:
            prompts = json.load(file)

        for i, prompt in enumerate(prompts):
            source = open(paths[images_names.index(image_name)].strip(), "rb").read()
            mask_file_path = os.path.join(args.masks_path, f"mask_{image_name}_{suffix}.png")
            mask = open(mask_file_path, "rb").read()
            
            replacement_idx = prompt.find("Replace with: ")
            if replacement_idx == -1:
                print(f"Unable to find the replacement in the image {image_name}_{suffix} for prompt {i}")
            else:
                raw_target = prompt[replacement_idx+len("Replace with: "):].strip()

                for stop_word in ["Rationale:", "Visual Style:", "Justification:", "Justify:", "Note:"]:
                    if stop_word in raw_target:
                        raw_target = raw_target.split(stop_word)[0].strip()

                lines = [line.strip() for line in raw_target.split('\n') if line.strip()]
                prompt_clean = lines[0] if lines else raw_target

                prompt_clean = re.sub(r'\(.*?\)', '', prompt_clean)
                prompt_clean = prompt_clean.strip('"').strip("'").strip()
                prompt_clean = " ".join(prompt_clean.split())
                
                if prompt_clean.lower().startswith("none"):
                    continue

                prompt_lower = prompt.lower()
                bw_keywords = ["monochrome", "black and white", "black & white", "b&w", "grayscale", "greyscale", "monochromatic"]

                if "visual style: color" in prompt_lower:
                    is_monochrome = False
                elif "visual style: monochrome" in prompt_lower:
                    is_monochrome = True
                else:
                    has_bw_word = any(kw in prompt_lower for kw in bw_keywords)
                    has_negation = "not monochrome" in prompt_lower or "not black and white" in prompt_lower or "is not b&w" in prompt_lower
                    is_monochrome = has_bw_word and not has_negation

                #print(f"|\033[33m[DEBUG]\033[0m Face target: {image_name}_{suffix}| B&W ? \033[32m{is_monochrome}\033[0m | Prompt: {prompt_clean}\n")

                current_prompt = prompt_clean

                if is_monochrome:
                    if "black and white" not in current_prompt.lower():
                        current_prompt = f"{current_prompt}, black and white photograph, monochrome, film grain, soft contrast, period-accurate photo texture"
                    negative_prompt = ("color, colorful, multicolored, vibrant, saturated, "
                                        "distorted, bad anatomy, oversaturated, digital sharpness")
                    style_selections = ["Misc Monochrome", "Photo Film Noir"]
                else:
                    current_prompt = f"{current_prompt}, period-accurate photo texture"
                    negative_prompt = ("distorted, deformed, low quality, bad anatomy, "
                                        "oversaturated")
                    style_selections = ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]

                # Salva il file includendo l'identificativo del volto corrente
                save_name = f"{image_name}_{suffix}_{i%5}_inpainted"

                result = inpaint_outpaint(
                    params={
                        "prompt": current_prompt,
                        "negative_prompt": negative_prompt,
                        "style_selections": style_selections,
                        "async_process": False, 
                        "save_name": f"{args.save_path}/{save_name}",
                        "guidance_scale": GUIDANCE_SCALE,
                        "inpaint_strength": INPAINT_STRENGTH,
                        "image_seed": IMAGE_SEED,
                    },
                    input_image=source,
                    input_mask=mask)

                # la API di Fooocus aggiunge "-0" al save_name
                file_name = f"{save_name}-0.png"
                with open(clean_prompts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "file_name": file_name,
                        "prompt": current_prompt,
                        "negative_prompt": negative_prompt,
                        "style_selections": style_selections,
                        "is_monochrome": is_monochrome,
                        "mask_path": mask_file_path,
                    }, ensure_ascii=False) + "\n")

    print(f"Inpainting ended in {(time()-start)/60} minutes.")
    
if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
