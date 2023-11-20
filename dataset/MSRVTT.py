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
    def __init__(self, path_to_metadata):
        with open(path_to_metadata, "r") as f:
            self.metadata = json.load(f)


    def __len__(self):
        return len(self.metadata["sentences"])


    def __getitem__(self, idx):
        video_id = int(self.metadata["sentences"][idx]["video_id"][5:])
        question = self.metadata["sentences"][idx]["caption"]
    
        return video_id, question