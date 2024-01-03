import json
import os
import torch

from args import parse_args
from dataset.MSRVTT import MSRVTT_Ranking_Dataset
from tqdm import tqdm
from transformers import DPRReader, DPRReaderTokenizerFast


def main(args):
    with open(args.path_to_titles, "r") as f:
        titles_dict = json.load(f)

    caption_path_list = [os.path.join(args.path_to_captions, x) for x in os.listdir(args.path_to_captions)]
    caption_path_list = sorted(caption_path_list)[:10]  # TODO

    assert(len(titles_dict) == len(caption_path_list))
    assert(set(range(len(titles_dict))) == set([int(x) for x in titles_dict]))

    num_videos = len(titles_dict)

    titles = [""] * num_videos
    texts = [""] * num_videos

    for video_id in titles_dict:
        titles[int(video_id)] = titles_dict[video_id]

    for caption_path in caption_path_list:
        video_id = int(os.path.splitext(os.path.basename(caption_path))[0])

        with open(caption_path, "r") as f:
            texts[video_id] = "\n".join(f.readlines()[:5])     # TODO


    msrvtt_train_dataset = MSRVTT_Ranking_Dataset(os.path.join(args.path_to_MSRVTT, "train_val_videodatainfo.json"))

    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

    tokenizer = DPRReaderTokenizerFast.from_pretrained("facebook/dpr-reader-single-nq-base")
    model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(device)

    for _, question in tqdm(msrvtt_train_dataset):
        relevance_logits = []

        for i in range(num_videos):
            encoded_inputs = tokenizer(
                questions = [question],
                titles = titles[i],
                texts = texts[i],
                return_tensors = "pt",
                padding = True,
                truncation = True
            ).to(device)
 
            outputs = model(**encoded_inputs)
            relevance_logits.append(outputs.relevance_logits[0].detach().cpu().numpy())
        
        retrieved_videos = sorted(range(len(relevance_logits)), key=lambda i: relevance_logits[i], reverse=True)[:args.top_n_retrieval]
        print(retrieved_videos)

if __name__ == "__main__":
    args = parse_args()
    main(args)