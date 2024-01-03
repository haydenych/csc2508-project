import json
import numpy as np
import os
import torch

from args import parse_args
from dataset.ActivityNet import ActivityNet_Ranking_Dataset
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast, DPRContextEncoder, DPRContextEncoderTokenizer, DPRContextEncoderTokenizerFast

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

    caption_path_list = [os.path.join(args.path_to_ActivityNet_captions, x) for x in os.listdir(args.path_to_ActivityNet_captions)]

    corpus = {}

    for caption_path in caption_path_list:
        video_id = os.path.splitext(os.path.basename(caption_path))[0]
        corpus[video_id] = []

        with open(caption_path, "r") as f:
            lines = f.readlines()

            # Split texts into 5 sentences
            dpr_window_size = 5
            for i in range(0, len(lines), dpr_window_size):
                corpus[video_id].append("".join(lines[i:i+dpr_window_size]))

    activitynet_dataset = ActivityNet_Ranking_Dataset(os.path.join(args.path_to_ActivityNet, "train.json"), args.path_to_ActivityNet_captions)

    tokenizer_context = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model_context = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

    # Some optimization
    if not os.path.exists(args.path_to_passage_embeddings):
        passage_embeddings = []
        passage_embedding_ids = []
        for video_id, texts in tqdm(corpus.items()):
            for text_window in texts:
                input_ids = tokenizer_context(text_window, return_tensors="pt")["input_ids"].to(device)
                embeddings = model_context(input_ids).pooler_output 

                passage_embeddings.append(embeddings.view(-1).detach().cpu().numpy())
                passage_embedding_ids.append(video_id)

        passage_embeddings = np.array(passage_embeddings)
        passage_embedding_ids = np.array(passage_embedding_ids)

        if not os.path.exists(os.path.dirname(args.path_to_passage_embeddings)):
            os.makedirs(os.path.dirname(args.path_to_passage_embeddings))

        with open(args.path_to_passage_embeddings, 'wb') as f:
            np.savez(f, passage_embeddings=passage_embeddings, passage_embedding_ids=passage_embedding_ids)

    else:
        with open(args.path_to_passage_embeddings, 'rb') as f:
            npz = np.load(f)
            passage_embeddings = npz["passage_embeddings"]
            passage_embedding_ids = npz["passage_embedding_ids"]

    # Query
    tokenizer_question = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model_question = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

    success = 0

    query_embeddings = []
    ground_truth = []
    if not os.path.exists(args.path_to_query_embeddings):
        for video_id_gt, query, _ in tqdm(activitynet_dataset):
            input_ids = tokenizer_question(query, return_tensors="pt")["input_ids"].to(device)
            query_embeddings.append(model_question(input_ids).pooler_output.view(-1).detach().cpu().numpy())
            ground_truth.append(video_id_gt)

        query_embeddings = np.array(query_embeddings)
        ground_truth = np.array(ground_truth)

        with open(args.path_to_query_embeddings, 'wb') as f:
            np.savez(f, query_embeddings=query_embeddings, ground_truth=ground_truth)

    else:
        with open(args.path_to_query_embeddings, 'rb') as f:
            npz = np.load(f)
            query_embeddings = npz["query_embeddings"]
            ground_truth = npz["ground_truth"]

    for i in tqdm(range(query_embeddings.shape[0])):
        sim_score = np.dot(passage_embeddings, query_embeddings[i, ...])
        retrieved_video_ids = passage_embedding_ids[np.argpartition(sim_score, -args.top_n_retrieval)[-args.top_n_retrieval: ]]

        if ground_truth[i] in retrieved_video_ids:
            success += 1

    print(f"Accuracy: {success / len(activitynet_dataset)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)