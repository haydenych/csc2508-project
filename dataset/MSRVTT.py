import cv2
import json
import numpy as np
import os
import torch

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

class MSRVTT_Captioning_Dataset(Dataset):
    def __init__(self, path_to_video_folder, capture_every):
        self.path_to_video_folder = path_to_video_folder
        self.video_path_list = [os.path.join(self.path_to_video_folder, x) for x in os.listdir(self.path_to_video_folder)]

        self.capture_every = capture_every


    def __len__(self):
        return len(self.video_path_list)


    def __getitem__(self, idx):
        video_id = int(os.path.splitext(os.path.basename(self.video_path_list[idx]))[0][5:])

        cap = cv2.VideoCapture(self.video_path_list[idx])
        frames = []

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
                frame_count += self.capture_every
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            else:
                cap.release()
                break

        return video_id, frames
    

class MSRVTT_Ranking_Dataset(Dataset):
    def __init__(self, path_to_metadata, path_to_title, path_to_captions_folder):
        self.path_to_metadata = path_to_metadata

        self.path_to_title = path_to_title

        self.path_to_captions_folder = path_to_captions_folder
        self.caption_path_list = [os.path.join(self.path_to_captions_folder, x) for x in os.listdir(self.path_to_captions_folder)]


    def __len__(self):
        return len(self.caption_path_list)


    def __getitem__(self, idx):
        video_id = int(os.path.splitext(os.path.basename(self.caption_path_list[idx]))[0])

        questions = []
        with open(self.path_to_metadata, "r") as f:
            metadata = json.load(f)

            for sentence in metadata["sentences"]:
                if sentence["video_id"] == f"video{video_id}":
                    questions.append(sentence["caption"])
    
        with open(self.path_to_title, "r") as f:
            title = json.load(f)[video_id]

        with open(self.caption_path_list[idx], "r") as f:
            lines = f.readlines()
            text = "\n".join(lines)

        return video_id, questions, title, text