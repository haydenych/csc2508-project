import os

from args import parse_args
from dataset.MSRVTT import MSRVTT_Ranking_Dataset
from tqdm import tqdm
from transformers import DPRReader, DPRReaderTokenizer


def main(args):
    msrvtt_train_dataset = MSRVTT_Ranking_Dataset(os.path.join(args.path_to_MSRVTT, "train_val_videodatainfo.json"),
                                                  args.path_to_title,
                                                  args.path_to_captions_folder
    )

    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

    tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
    model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(device)

    for video_id, questions, title, text in tqdm(msrvtt_train_dataset):
        for question in questions:
            encoded_inputs = tokenizer(
                questions=[question] * 1000,
                titles=["Haddaway"] * 1000,
                texts=["'What Is Love' is a song recorded by the artist Haddaway"] * 1000,
                return_tensors="pt",
                padding=True,
                truncation=True
            )


    outputs = model(**encoded_inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    relevance_logits = outputs.relevance_logits

    # print(start_logits)

    # print(end_logits)

    print(relevance_logits)

if __name__ == "__main__":
    args = parse_args()
    main(args)