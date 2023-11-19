import json
import os
import requests

from args import parse_args
from bs4 import BeautifulSoup
from tqdm import tqdm


def main(args):
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

    with open(args.path_to_titles, "w") as f:
        json.dump(video_titles, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)