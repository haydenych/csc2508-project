import json
import os
import torch

from args import parse_args
from dataset.MSRVTT import MSRVTT_Ranking_Dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import DPRReader, DPRReaderTokenizer


def main(args):
    with open(args.path_to_titles, "r") as f:
        titles_dict = json.load(f)

    caption_path_list = [os.path.join(args.path_to_captions_folder, x) for x in os.listdir(args.path_to_captions_folder)]

    assert(len(titles_dict) == len(caption_path_list))
    assert(set(range(len(titles_dict))) == set([int(x) for x in titles_dict]))

    num_videos = len(titles_dict)

    titles = [""] * num_videos
    corpus = [""] * num_videos

    for video_id in titles_dict:
        titles[int(video_id)] = titles_dict[video_id]

    for caption_path in caption_path_list:
        video_id = int(os.path.splitext(os.path.basename(caption_path))[0])

        with open(caption_path, "r") as f:
            corpus[video_id] = titles[video_id] + "\n" + "".join(f.readlines())

    inverted_corpus = {}
    for i, doc in enumerate(corpus):
        inverted_corpus[doc] = i

    tokenized_corpus = [doc.replace("\n", " ").split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    msrvtt_train_dataset = MSRVTT_Ranking_Dataset(os.path.join(args.path_to_MSRVTT, "train_val_videodatainfo.json"))

    success = 0

    for video_id_gt, query in tqdm(msrvtt_train_dataset):
        tokenized_query = query.split(" ")
        retrieved_videos = bm25.get_top_n(tokenized_query, corpus, n=args.top_n_retrieval)
        retrieved_video_ids = [inverted_corpus[video] for video in retrieved_videos]

        if video_id_gt in retrieved_video_ids:
            success += 1

    print(f"Accuracy: {success / len(msrvtt_train_dataset)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)