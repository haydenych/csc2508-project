import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help="GPU Device ID, -1 for CPU")
    parser.add_argument('--path-to-MSRVTT', type=str, default='../data/MSRVTT', help="Path to MSRVTT dataset folder")
    parser.add_argument('--path-to-captions_folder', type=str, default='../data/MSRVTT/Captions_Train', help="Path to video frame captions")
    parser.add_argument('--path-to-titles', type=str, default='../data/MSRVTT/titles_train.json', help="Path to video titles")

    parser.add_argument('--capture-every', type=int, default=25, help="Capture every <value> frame, default: 25")
    parser.add_argument('--top-n-retrieval', type=int, default=20, help="Number of videos to retrieve")

    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--n-epochs', type=int, default=20, help="Number of epochs")

    parser.add_argument('--verbose', type=bool, default=False, help="Debug mode")

    args = parser.parse_args()
    return args