import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help="GPU Device ID, -1 for CPU")
    parser.add_argument('--path-to-MSRVTT', type=str, default='../../../data/MSRVTT', help="Path to MSRVTT dataset folder")
    parser.add_argument('--path-to-MSRVTT-captions', type=str, default='../../../data/MSRVTT/captions2_train', help="Path to video frame captions")
    parser.add_argument('--path-to-MSRVTT-titles', type=str, default='../../../data/MSRVTT/titles_train.json', help="Path to video titles")
    
    parser.add_argument('--path-to-ActivityNet', type=str, default='../../../data/captions', help="Path to ActivityNet dataset folder")
    # parser.add_argument('--path-to-ActivityNet-captions', type=str, default='../../../data/captions/captions2_train_20', help="Path to video frame captions")
    parser.add_argument('--path-to-ActivityNet-captions', type=str, default='../../../data/captions/captions2_train_5_20_combined', help="Path to video frame captions")

    parser.add_argument('--path-to-passage-embeddings', type=str, default="./output/passage_embeddings_5_20.npy")
    parser.add_argument('--path-to-query-embeddings', type=str, default="./output/query_embeddings_activitynet.npy")

    # parser.add_argument('--path-to-logs', type=str, default='./logs/MSRVTT', help="Path to logs")
    parser.add_argument('--path-to-logs', type=str, default='./logs/ActivityNet', help="Path to logs")


    parser.add_argument('--chunk-size', type=int, default=5, help="Size of each video chunk in seconds")

    parser.add_argument('--top-n-retrieval', type=int, default=100, help="Number of videos to retrieve")

    parser.add_argument('--resume', type=bool, default=True, help="Continue from existing files")

    args = parser.parse_args()
    return args