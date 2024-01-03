import cv2
import math
import numpy as np
import os
import torch

from args import parse_args
from dataset.ActivityNet import ActivityNet_Captioning_Dataset
from datetime import datetime
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


def main(args):
    if not os.path.exists(args.path_to_ActivityNet_captions):
        os.makedirs(args.path_to_ActivityNet_captions)

    if not os.path.exists(args.path_to_logs):
        os.makedirs(args.path_to_logs)

    log_filename = os.path.join(args.path_to_logs, datetime.now().strftime("%d%m%Y_%H%M%S") + ".txt")

    activitynet_train_dataset = ActivityNet_Captioning_Dataset(os.path.join(args.path_to_ActivityNet, "videos/train"))

    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    for video_id, cap in tqdm(activitynet_train_dataset):
        if args.resume:
            if os.path.exists(os.path.join(args.path_to_ActivityNet_captions, f"{video_id}.txt")):
                continue

        try:
            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = round(cap.get(cv2.CAP_PROP_FPS))

            output_filename = os.path.join(args.path_to_ActivityNet_captions, f"{video_id}.txt")

            captions = []
            for i in range(math.ceil(num_frames / (args.chunk_size*fps))):
                indices = set(np.linspace(0, args.chunk_size*fps, num=model.config.encoder.num_frames, endpoint=False).astype(np.int64))

                if i == math.ceil(num_frames / (args.chunk_size*fps)) - 1:
                    indices = set(np.linspace(0, num_frames % (args.chunk_size*fps), num=model.config.encoder.num_frames, endpoint=False).astype(np.int64))
                
                frames = []
                for j in range(args.chunk_size*fps):
                    ret, frame = cap.read()

                    if ret:
                        if j in indices:
                            frames.append(frame)
                    else:
                        break

                # Discard remaining frames if there are less than the number of frames needed by the model (8)
                if len(frames) < model.config.encoder.num_frames:
                    continue

                # generate caption
                gen_kwargs = {
                    "min_length": 10, 
                    "max_length": 100, 
                    "num_beams": 8,
                }

                pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
                tokens = model.generate(pixel_values, **gen_kwargs)
                caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

                captions.append(caption)

            with open(output_filename, "w") as f:
                for line in captions:
                    f.write(f"{line}\n")

            cap.release()

        except:
            with open(log_filename, "a") as f:
                    f.write(f"{video_id}\n")
                

if __name__ == "__main__":
    args = parse_args()
    main(args)