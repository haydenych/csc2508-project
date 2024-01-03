import cv2
import json
import os

from torch.utils.data.dataset import Dataset

class ActivityNet_Captioning_Dataset(Dataset):
    def __init__(self, path_to_videos):
        self.path_to_videos = path_to_videos
        self.video_path_list = sorted([os.path.join(self.path_to_videos, x) for x in os.listdir(self.path_to_videos)])


    def __len__(self):
        return len(self.video_path_list)


    def __getitem__(self, idx):
        video_id = os.path.splitext(os.path.basename(self.video_path_list[idx]))[0]
        cap = cv2.VideoCapture(self.video_path_list[idx])

        return video_id, cap
    

class ActivityNet_Ranking_Dataset(Dataset):
    def __init__(self, path_to_metadata, path_to_captions):
        with open(path_to_metadata, "r") as f:
            self.metadata = json.load(f)

        # I dont know why they have mismatch ids across files
        captioned_ids = sorted([os.path.splitext(os.path.basename(x))[0] for x in os.listdir(path_to_captions)])
        metadata_ids = list(self.metadata.keys())
        ids = list(set(captioned_ids).intersection(set(metadata_ids)))

        self.ids = []
        self.queries = []
        self.timestamps = []

        for video_id in ids:
            assert(len(self.metadata[video_id]["sentences"]) == len(self.metadata[video_id]["timestamps"]))

            num_queries = len(self.metadata[video_id]["sentences"])
            self.ids.extend([video_id] * num_queries) 
            self.queries.extend(self.metadata[video_id]["sentences"])
            self.timestamps.extend(self.metadata[video_id]["timestamps"])


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        video_id = self.ids[idx]
        query = self.queries[idx]
        timestamp = self.timestamps[idx]

        return video_id, query, timestamp
    
