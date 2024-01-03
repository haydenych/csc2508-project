# CSC2508 Project

This repository includes for the CSC2508 Project

# Dataset
Download the ActivityNet Captions Dataset [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/). 

After downloading, `cd preprocess` and run the script `python3 activitynet_captions.py --path-to-data=<PATH>` to download the videos from YouTube.

# Setup
To setup the conda environment, run `conda env create -f env.yml` and then activate the environment using `conda activate CSC2508_Project`.

Since this code is run on a linux environment, there might be problems when setting up on MacBook, in particular with the libc files.

# Scripts

```
cd src/experiments
```

To generate the video captions, run

```
python3 captioning-activitynet.py --path-to-ActivityNet=<PATH> --path-to-ActivityNet-captions=<PATH> --chunksize=5
```

To perform document retrieval, run the following depending on which retrieval method you would like to use (BM25 / FAISS / DPR)
```
python3 ranking-activitynet-bm25.py --path-to-ActivityNet=<PATH> --path-to-ActivityNet-captions=<PATH> --top-n-retrieval=1
python3 ranking-activitynet-faiss.py --path-to-ActivityNet=<PATH> --path-to-ActivityNet-captions=<PATH> --top-n-retrieval=1
python3 ranking-activitynet-dpr.py --path-to-ActivityNet=<PATH> --path-to-ActivityNet-captions=<PATH> --top-n-retrieval=1
```