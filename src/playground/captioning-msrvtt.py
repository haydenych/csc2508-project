import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import torch

from args import parse_args
from dataset.MSRVTT import MSRVTT_Captioning_Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration

def main(args):
    msrvtt_train_dataset = MSRVTT_Captioning_Dataset(os.path.join(args.path_to_MSRVTT, "TrainValVideo"))

    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    for video_id, frames, fps in tqdm(msrvtt_train_dataset):
        text = "A picture of"
        inputs = processor(images=frames, text=text, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        output_filename = os.path.join(args.path_to_captions, f"{video_id}.txt")

        with open(output_filename, "w") as f:
            frame_count = 0

            for line in generated_captions:
                f.write(f"Frame {frame_count}: {line}\n")
                frame_count += fps


if __name__ == "__main__":
    args = parse_args()
    main(args)