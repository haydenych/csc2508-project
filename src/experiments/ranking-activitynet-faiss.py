import faiss
import numpy as np
import os

from args import parse_args
from dataset.ActivityNet import ActivityNet_Ranking_Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def main(args):
    caption_path_list = [os.path.join(args.path_to_ActivityNet_captions, x) for x in os.listdir(args.path_to_ActivityNet_captions)]

    corpus = []
    video_ids = []
    for caption_path in caption_path_list:
        video_id = os.path.splitext(os.path.basename(caption_path))[0]

        with open(caption_path, "r") as f:
            doc = "".join(f.readlines())
            corpus.append(doc)
            video_ids.append(video_id)

    video_ids = np.array(video_ids)

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = encoder.encode(corpus)

    index = faiss.IndexFlatL2(vectors.shape[1])
    faiss.normalize_L2(vectors)
    index.add(vectors)

    activitynet_train_dataset = ActivityNet_Ranking_Dataset(os.path.join(args.path_to_ActivityNet, "train.json"), args.path_to_ActivityNet_captions)

    success = 0

    for video_id_gt, query, _ in tqdm(activitynet_train_dataset):
        query_vector = encoder.encode(query)
        query_vector = np.array([query_vector])

        faiss.normalize_L2(query_vector)

        dist, ann = index.search(query_vector, args.top_n_retrieval)
        retrieved_video_ids = video_ids[ann[0]]
        
        if video_id_gt in retrieved_video_ids:
            success += 1

    print(f"Accuracy: {success / len(activitynet_train_dataset)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)