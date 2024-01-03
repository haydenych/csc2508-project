import cv2
import json
import numpy as np
import os
import torch

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

class MSRVTT_Captioning_Dataset(Dataset):
    def __init__(self, path_to_video_folder):
        self.path_to_video_folder = path_to_video_folder
        self.video_path_list = sorted([os.path.join(self.path_to_video_folder, x) for x in os.listdir(self.path_to_video_folder)])


    def __len__(self):
        return len(self.video_path_list)


    def __getitem__(self, idx):
        video_id = int(os.path.splitext(os.path.basename(self.video_path_list[idx]))[0][5:])
        cap = cv2.VideoCapture(self.video_path_list[idx])

        return video_id, cap
    

class MSRVTT_Ranking_Dataset(Dataset):
    def __init__(self, path_to_metadata):
        with open(path_to_metadata, "r") as f:
            self.metadata = json.load(f)


    def __len__(self):
        return len(self.metadata["sentences"])


    def __getitem__(self, idx):
        video_id = int(self.metadata["sentences"][idx]["video_id"][5:])
        query = self.metadata["sentences"][idx]["caption"]
    
        return video_id, query