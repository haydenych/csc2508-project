import argparse
import json
import os

from logger import Logger
from pytube import YouTube
from tqdm import tqdm


def download(ids, output_dir, logger):
    for id in tqdm(ids):
        try:
            yt = YouTube(f"https://youtube.com/watch?v={id[2:]}")
            yt.streams \
                .filter(progressive=True, file_extension="mp4") \
                .order_by("resolution") \
                .desc() \
                .first() \
                .download(output_path=output_dir, filename=f"{id}.mp4")

        except:
            logger.log(f"Failed to download id: {id}")


def main(args):
    logger = Logger(args.path_to_log)


    # Download Train Videos
    with open(os.path.join(args.path_to_data, "train_ids.json")) as f:
        train_ids = json.load(f)

    train_output_dir = os.path.join(args.path_to_data, "videos/train")

    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)

    logger.log("Downloading Train Videos")
    logger.log(f"Number of Videos: {len(train_ids)}")
    download(train_ids, train_output_dir, logger)
    logger.log()
    logger.log()


    # Download Validate Videos
    with open(os.path.join(args.path_to_data, "val_ids.json")) as f:
        val_ids = json.load(f)

    val_output_dir = os.path.join(args.path_to_data, "videos/val")

    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    logger.log("Downloading Validate Videos")
    logger.log(f"Number of Videos: {len(val_ids)}")
    download(val_ids, val_output_dir, logger)
    logger.log()
    logger.log()


    # Download Test Videos
    with open(os.path.join(args.path_to_data, "test_ids.json")) as f:
        test_ids = json.load(f)

    test_output_dir = os.path.join(args.path_to_data, "videos/test")

    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    logger.log("Downloading Test Videos")
    logger.log(f"Number of Videos: {len(test_ids)}")
    download(test_ids, test_output_dir, logger)
    logger.log()
    logger.log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-to-data", type=str, default="../../data/captions")
    parser.add_argument("--path_to_log", type=str, default="./logs")

    args = parser.parse_args()

    main(args)