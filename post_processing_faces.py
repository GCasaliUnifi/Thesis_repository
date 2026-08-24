#!/usr/bin/env python3

import numpy as np
import json
from PIL import Image
from time import time
import os
import random
import cv2
from tqdm import tqdm
import argparse

def sample_n_images(input_dir, n_images, dataset_name):
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("json")]
    if n_images is None or n_images > len(json_files):
        print(f"You are using all the files in the input directory.")
        with open(f'./sampled_{len(json_files)}_images_{dataset_name}.txt', 'w') as f:
            for file in json_files:
                f.write(file+"\n")
        return json_files
    else:
        json_files_sampled = random.sample(json_files, n_images)
        with open(f'./sampled_{n_images}_images_{dataset_name}.txt', 'w') as f:
            for file in json_files_sampled:
                f.write(file+"\n")
        return json_files_sampled


def compute_mask_area(masks, box):
    counts = np.asarray([0,0,0], dtype=np.float32)
    masks = masks.astype(np.uint8)
    for chn in range(masks.shape[1]):
        counts[chn] = masks[box][chn].sum()
                    
    areas = counts / (masks.shape[2]*masks.shape[3])
    max_area = areas.max()
    chn = areas.argmax()

    return float(max_area), int(chn)

def compute_connected_components(image):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(image)
    num_connected_components = num_labels - 1  
    return num_connected_components

def update_json_file(data, image_json_path):
    masks = np.load(image_json_path.replace(".json", ".npz"))['masks']
    
    for box_num, box in enumerate(data['boxes']):
        area, chn = compute_mask_area(masks, box_num)
        
        box['area'] = area
        if 0 <= area and area <= 0.15:
            dim = 'small'
        elif 0.15 < area and area <= 0.3:
            dim = 'medium'
        elif 0.3 < area and area <= 0.6:
            dim = 'large'
        else:
            dim = 'extra_large'
            
        box['area_size'] = dim
        box['channel_max_area'] = chn
        
        image_array = masks[box_num, chn, :, :].astype(np.uint8) * 255
        num_connected_components = compute_connected_components(image_array)
        box['num_connected_components'] = num_connected_components
    
    with open(image_json_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_images_path(sampled_json_files, dataset_name):
    with open(sampled_json_files, 'r') as f:
        json_files = f.readlines()
    
    images_txt_file = open(f"./source_images_path_{dataset_name}.txt", 'w')
    
    for json_file in json_files:
        with open(json_file.strip(), 'r') as f:
            data = json.load(f)
        images_txt_file.write(data['path']+"\n")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="Directory containing the output from the mask extraction component (.json and .npz files).")
    parser.add_argument('--num_images', type=int, default=None, help="Number of samples to construct the BtB collection for the pipeline.")
    parser.add_argument('--dataset_name', type=str, default="FacesDataset", help="Name of the source dataset used.")
    parser.add_argument('--save_dir_masks', type=str, help="Directory to save the created masks.")
    parser.add_argument('--save_dir_bb', type=str, help="Directory to save the images with the green bounding box.")
    return parser 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    masks_save_path = args.save_dir_masks
    os.makedirs(masks_save_path, exist_ok=True)
    
    bounding_box_save_path = args.save_dir_bb
    os.makedirs(bounding_box_save_path, exist_ok=True)
    
    best_mask_labels_file = open(f'./best_mask_labels_{args.dataset_name}.txt', 'a')
    
    json_files_sampled = sample_n_images(args.input_dir, args.num_images, args.dataset_name)
    
    for json_file in tqdm(json_files_sampled, desc="Processing faces", bar_format=f'\033[35m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
        json_file = json_file.strip()
        
        with open(json_file, 'r') as file:
            data = json.load(file)
            
        image_name = data['path'].split("/")[-1].replace(".jpg", "").replace(".png", "")
        
        update_json_file(data, json_file)
        
        masks = np.load(json_file.replace('.json', '.npz'))['masks']
        original_img = cv2.imread(data['path'])
        
        if len(data['boxes']) == 0:
            continue  #ssalta immagine se DINO non ha trovato volti
            
        for idx, box in enumerate(data['boxes']):
            suffix = f"face{idx}"  # Esempio: 0000001_face0, 0000001_face1
            chn = box['channel_max_area']
            
            image_array = masks[idx, chn, :, :].astype(np.uint8) * 255
            
            # kernel ridotto (per i dettagli dei volti piccoli)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            opening = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            cv2.imwrite(os.path.join(masks_save_path, f'mask_{image_name}_{suffix}.png'), closing)
            
            best_mask_labels_file.write(f"{image_name}_{suffix} {box['label']}\n")
            
            start_point = (box['box']['xmin'], box['box']['ymin'])
            end_point = (box['box']['xmax'], box['box']['ymax'])
            
            img_with_box = original_img.copy()
            cv2.rectangle(img_with_box, start_point, end_point, color=(0, 255, 0), thickness=3)
            
            cv2.imwrite(os.path.join(bounding_box_save_path, f'mask_{image_name}_{suffix}_with_box.png'), img_with_box)

    # 4. CREATE A TXT FILE CONTAINING THE SOURCE IMAGES PATH
    get_images_path(f'./sampled_{len(json_files_sampled)}_images_{args.dataset_name}.txt', args.dataset_name)
    best_mask_labels_file.close()
        
    print(f"Dataset {args.dataset_name} face post-processing completed.")
