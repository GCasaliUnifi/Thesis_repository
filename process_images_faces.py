#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
import gc
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMaskGeneration, AutoModelForZeroShotObjectDetection, pipeline
from ram.models import ram
from ram import get_transform


class Detector:
    def __init__(self, device):
        model_id = "IDEA-Research/grounding-dino-tiny"

        self.object_detector = pipeline(model=model_id, task="zero-shot-object-detection", device=device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def __call__(self, images):
        inputs = [
            {
                "image": i,
                "candidate_labels": ["face."]
            }
            for i in images
        ]
        return self.object_detector(inputs)


class Segmenter:
    def __init__(self, device, batch_boxes=2): # MODIFICATO da 4 a 2
        segmenter_id = "facebook/sam-vit-base"

        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        self.processor = AutoProcessor.from_pretrained(segmenter_id)
        self.device = device
        self.batch_boxes = batch_boxes

    def __call__(self, image, boxes):
        n_batches = len(boxes) // self.batch_boxes + (1 if len(boxes) % self.batch_boxes != 0 else 0)

        all_masks = np.empty((len(boxes), 3, image.size[1], image.size[0]), dtype=bool)
        
        print(f"Running {n_batches} batches")
        with torch.no_grad():
            for i in range(n_batches):
                input_boxes = [boxes[i * self.batch_boxes : (i + 1) * self.batch_boxes]]
                inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)

                outputs = self.segmentator(**inputs)
                masks = self.processor.post_process_masks(
                    masks=outputs.pred_masks,
                    original_sizes=inputs.original_sizes,
                    reshaped_input_sizes=inputs.reshaped_input_sizes
                )[0]
    
                all_masks[i * self.batch_boxes : min((i + 1) * self.batch_boxes, len(boxes)), :, :, :] = masks.cpu().numpy()
        
        return all_masks


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-o', '--output-path', default='./out')
    parser.add_argument('-t', '--top-masks', type=int, default=None)
    parser.add_argument('input_list_path')

    return parser

class Batched:
    def __init__(self, base, batch_size):
        self.base = base
        self.batch_size = batch_size
        self.next_batch = 0

    def __len__(self):
        base_len = len(self.base)
        if base_len % self.batch_size != 0:
            return base_len // self.batch_size + 1
        else:
            return base_len // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch >= len(self):
            raise StopIteration

        items = self.base[self.next_batch * self.batch_size : (self.next_batch + 1) * self.batch_size]
        self.next_batch += 1

        return items


def load_all_json(parent_path, concat=False):
    result = []
    for item in sorted(os.listdir(parent_path)):
        with open(os.path.join(parent_path, item), 'r') as stream:
            result.append(json.load(stream))

    if concat:
        return [item for sub in result for item in sub]
    else:
        return result


def main(args):
    detector = Detector(args.device)

    with open(args.input_list_path, 'r') as stream:
        image_paths = [x.strip() for x in stream.readlines()]

    os.makedirs(os.path.join(args.output_path, 'tmp', 'boxes'), exist_ok=True)
    
    for i, ps in enumerate(tqdm(Batched(image_paths, 16))):
        boxes_path = os.path.join(args.output_path, 'tmp', 'boxes', f'{i:07d}-boxes.json')
        if not os.path.exists(boxes_path):
            images = [Image.open(p).convert("RGB") for p in ps]
            boxes = detector(images)
            
            with open(boxes_path, 'w') as stream:
                json.dump(boxes, stream)

    del detector
    gc.collect()
    torch.cuda.empty_cache()

    all_boxes = load_all_json(os.path.join(args.output_path, 'tmp', 'boxes'), concat=True)
    print("Boxes OK")

    segmenter = Segmenter(args.device)

    for i, (p, boxes) in enumerate(zip(tqdm(image_paths), all_boxes)):
        result_path = os.path.join(args.output_path, f'{i:07d}.json')
        if not os.path.exists(result_path):
            image = Image.open(p).convert("RGB")
            if args.top_masks is not None:
                boxes = boxes[:args.top_masks]
            in_boxes = [[m['box']['xmin'], m['box']['ymin'], m['box']['xmax'], m['box']['ymax']] for m in boxes]
            segmented_masks = segmenter(image, in_boxes)
            
            result = {
                'path': p,
                'tags': ["face"], #provo con tag fisso per compatibilità
                'boxes': [
                    { 'score': box['score'], 'label': box['label'][:-1], 'box': box['box'] }
                    for box in boxes
                ]
            }
            
            with open(result_path, 'w') as stream:
                json.dump(result, stream, indent=2)

            masks_path = os.path.join(args.output_path, f'{i:07d}.npz')
            np.savez_compressed(masks_path, masks=segmented_masks)
    
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
