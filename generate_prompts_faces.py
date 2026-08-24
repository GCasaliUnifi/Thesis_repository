import os
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import json
from tqdm import tqdm
import re
from time import time

from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def main(args):
    # LOAD THE VISUAL LANGUAGE MODEL 
    model_path = 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(num_beams=1,max_new_tokens=512, do_sample=True)
    
    # LOAD THE TXT FILE WITH THE LABELS FOR EACH IMAGE AND THE INPUT IMAGES
    with open(args.labels_file, 'r') as f:
        images_and_labels = f.readlines()
    imgs = [img.split(" ")[0] for img in images_and_labels]
    labels = [img.split(" ")[1] for img in images_and_labels]  

    images = os.listdir(args.bounding_box_dir)
    images = [image.replace("mask_", "").replace("_with_box.png", "") for image in images]
    
    responses = []

    with torch.no_grad():
        for img in tqdm(images, desc="Generating prompts", bar_format=f'\033[34m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
            img_name = img
            label = labels[imgs.index(img_name)]
                    
            pixel_values = load_image(os.path.join(args.bounding_box_dir, f"mask_{img}_with_box.png"), max_num=5).to(torch.float16).cuda()
    
            question = f"You are an expert of image inpainting and portrait generation. In this image, the green bounding box highlights a human face (originally contextualized as {label}). Propose a single alternative character identity, new face appereance, facial expression, or portrait feature to replace it with.\nThe replacement face must perfectly match the surrounding context, body structure, gender presentation, and lighting of the scene.\n\nCRITICAL VISION TASK: Look closely at the image colors. If the image is vintage, greyscale, sepia or completely lacks distinct modern colors, classify it as 'Monochrome', in any other case classify it as 'Color'.\n\nYou MUST follow this output format strictly. Do not add other text:\nRationale: [Short reason for the choice]\nVisual Style: [Write 'Monochrome' or 'Color']\nReplace with: [Descrive the new face/expression, max 4 words]"

            prompts=[]
        
            # PROMPT GENERATION
        
            for i in range(args.num_prompt):
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                responses.append(f"{img_name} - {response}")
                prompts.append(f"{response}")
    
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, f"{img_name}.json"), 'w') as out_file:
                    json.dump(prompts, out_file, indent=2)

            torch.cuda.empty_cache()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Directory to store the json file with the generated prompt for each image.')
    parser.add_argument('--bounding_box_dir', type=str, help='Directory containing the images with the green bounding box.')
    parser.add_argument('--labels_file', type=str, help='Txt file containing the labels for each image')
    parser.add_argument('--num_prompt', type=int, default=5, help='Number of prompts to generate for each image. Default is 5.')
    return parser 

if __name__ == '__main__':
    parser = get_parser()
    start = time()
    main(parser.parse_args())
    print(f"Prompt generation ended in {(time()-start)/60} minutes.")
