import argparse
import json
import os
import requests

from bs4 import BeautifulSoup
from logger import Logger
from tqdm import tqdm


def main(args):
    logger = Logger(args.path_to_log)

    MSRVTT
    with open(os.path.join(args.path_to_MSRVTT, "train_val_videodatainfo.json")) as f:
        train_metadata = json.load(f)

    video_titles = {}

    for video_info in tqdm(train_metadata["videos"]):
        r = requests.get(video_info["url"])
        soup = BeautifulSoup(r.text, features='lxml')

        link = soup.find_all(name="title")[0]
        title = str(link)
        title = title.replace("<title>","")
        title = title.replace("</title>","")

        video_titles[video_info["id"]] = title

    with open(args.path_to_MSRVTT_titles, "w") as f:
        json.dump(video_titles, f)


    # ActivityNet
    with open(os.path.join(args.path_to_ActivityNet, "train_ids.json")) as f:
        train_ids = json.load(f)

    video_titles = {}

    for train_id in tqdm(train_ids):
        try:
            r = requests.get(f"https://youtube.com/watch?v={train_id[2:]}")
            soup = BeautifulSoup(r.text, features='lxml')

            link = soup.find_all(name="title")[0]
            title = str(link)
            title = title.replace("<title>","")
            title = title.replace("</title>","")

            video_titles[train_id] = title

        except:
            logger.log(f"Failed to retrieve title: {train_id}")

    with open(args.path_to_ActivityNet_titles, "w") as f:
        json.dump(video_titles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path-to-MSRVTT', type=str, default='../../data/MSRVTT', help="Path to MSRVTT dataset folder")
    parser.add_argument('--path-to-ActivityNet', type=str, default='../../data/captions', help="Path to ActivityNet dataset folder")

    parser.add_argument('--path-to-MSRVTT-titles', type=str, default='../../data/MSRVTT/titles_train.json', help="Path to video titles")
    parser.add_argument('--path-to-ActivityNet-titles', type=str, default='../../data/captions/titles_train.json', help="Path to video titles")

    parser.add_argument('--path-to-log', type=str, default='./logs', help="Path to logs")

    args = parser.parse_args()
    main(args)

