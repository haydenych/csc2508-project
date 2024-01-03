import math
import os
import torch

from args import parse_args
from dataset.ActivityNet import ActivityNet_Captioning_Dataset
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration

def main(args):
    activitynet_train_dataset = ActivityNet_Captioning_Dataset(os.path.join(args.path_to_ActivityNet, "videos/train"))

    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    for video_id, frames, fps in tqdm(activitynet_train_dataset):
        BATCH_SIZE = 100

        generated_captions = []
        for i in range(math.ceil(len(frames) / BATCH_SIZE)):
            text = "A picture of"
            inputs = processor(images=frames[i*BATCH_SIZE:(i+1)*BATCH_SIZE], text=text, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values

            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_captions.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))

        if not os.path.exists(args.path_to_captions):
            os.makedirs(args.path_to_captions)

        output_filename = os.path.join(args.path_to_captions, f"{video_id}.txt")

        with open(output_filename, "w") as f:
            frame_count = 0

            for line in generated_captions:
                f.write(f"Frame {frame_count}: {line}\n")
                frame_count += fps


if __name__ == "__main__":
    args = parse_args()
    main(args)