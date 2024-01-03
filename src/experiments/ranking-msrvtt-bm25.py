import json
import os
import torch

from args import parse_args
from dataset.MSRVTT import MSRVTT_Ranking_Dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import DPRReader, DPRReaderTokenizer


def main(args):
    with open(args.path_to_MSRVTT_titles, "r") as f:
        titles_dict = json.load(f)

    caption_path_list = [os.path.join(args.path_to_MSRVTT_captions, x) for x in os.listdir(args.path_to_MSRVTT_captions)]

    corpus = []
    inverted_corpus = {}

    # Assumes documents are unique
    # This is not the case if we split the sentences into corpus
    for caption_path in caption_path_list:
        video_id = os.path.splitext(os.path.basename(caption_path))[0]

        with open(caption_path, "r") as f:
            doc = titles_dict[video_id] + "\n" + "".join(f.readlines())
            corpus.append(doc)
            inverted_corpus[doc] = int(video_id)

    tokenized_corpus = [doc.replace("\n", " ").split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    msrvtt_dataset = MSRVTT_Ranking_Dataset(os.path.join(args.path_to_MSRVTT, "train_val_videodatainfo.json"))

    success = 0
    tot = 0

    for video_id_gt, query in tqdm(msrvtt_dataset):
        tokenized_query = query.split(" ")
        retrieved_videos = bm25.get_top_n(tokenized_query, corpus, n=args.top_n_retrieval)
        print(query)
        print()
        print(retrieved_videos)
        print()
        print()
        print()
        retrieved_video_ids = [inverted_corpus[video] for video in retrieved_videos]

        if video_id_gt in retrieved_video_ids:
            success += 1
        
        tot += 1
        # print(success/tot)

    print(f"Accuracy: {success / len(msrvtt_dataset)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)