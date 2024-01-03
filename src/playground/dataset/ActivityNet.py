import cv2
import json
import os

from torch.utils.data.dataset import Dataset

class ActivityNet_Captioning_Dataset(Dataset):
    def __init__(self, path_to_videos):
        self.path_to_videos = path_to_videos
        self.video_path_list = [os.path.join(self.path_to_videos, x) for x in os.listdir(self.path_to_videos)]


    def __len__(self):
        return len(self.video_path_list)


    def __getitem__(self, idx):
        video_id = os.path.splitext(os.path.basename(self.video_path_list[idx]))[0]

        cap = cv2.VideoCapture(self.video_path_list[idx])
        frames = []

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
                frame_count += fps
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            else:
                cap.release()
                break

        return video_id, frames, fps
    

class ActivityNet_Ranking_Dataset(Dataset):
    def __init__(self, path_to_metadata):
        with open(path_to_metadata, "r") as f:
            self.metadata = json.load(f)


    def __len__(self):
        return len(self.metadata["sentences"])


    def __getitem__(self, idx):
        video_id = int(self.metadata["sentences"][idx]["video_id"][5:])
        question = self.metadata["sentences"][idx]["caption"]
    
        return video_id, question