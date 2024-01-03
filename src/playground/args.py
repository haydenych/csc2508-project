import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help="GPU Device ID, -1 for CPU")
    parser.add_argument('--path-to-MSRVTT', type=str, default='../../../data/MSRVTT', help="Path to MSRVTT dataset folder")
    parser.add_argument('--path-to-ActivityNet', type=str, default='../../../data/captions', help="Path to ActivityNet dataset folder")

    parser.add_argument('--path-to-captions', type=str, default='../../../data/MSRVTT/captions_train', help="Path to video frame captions")

    parser.add_argument('--path-to-titles', type=str, default='../../../data/MSRVTT/titles_train.json', help="Path to video titles")

    parser.add_argument('--path-to-passage-embeddings', type=str, default="./output/msrvtt_passage_embeddings.npy")

    parser.add_argument('--top-n-retrieval', type=int, default=20, help="Number of videos to retrieve")

    parser.add_argument('--verbose', type=bool, default=False, help="Debug mode")

    args = parser.parse_args()
    return args