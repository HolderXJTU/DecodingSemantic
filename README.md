# Decoding Semantic Concept Through Brain fMRI Spatio-Temporal Voxel Patterns


![Python](https://img.shields.io/badge/python-3.8%2B-blue)


## Contents

- [Overview](#overview)
- [Repository Contents](#repository-contents)
- [Environmental Requirements](#environmental-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [License](./LICENSE)
- [Issues](https://github.com/HolderXJTU/DecodingSemantic/issues)
- [Citation](#citation)

# Overview

Cognitive neuroscience bridges insights from human brain mechanisms to inspire artificial intelligence. Functional magnetic resonance imaging (fMRI) is widely utilized to investigate brain activity patterns, yet the modeling of cortical mechanisms for classifying semantic concepts embedded in visual stimuli remains underexplored. In this study, we designed a controlled cognitive experiment to analyze visual semantic processing, collecting fMRI data from 15 subjects. A spatio-temporal graph network was developed to capture dynamic features of semantic brain regions, enabling the construction of a functional network for semantic concept classification and prediction. Leveraging self-supervised learning, our decoding framework reconstructed visual stimuli and compared reconstructed images with predicted category outputs, ultimately deriving semantic representations. The experimental results highlight the superiority of the model in decoding the fMRI data, surpassing existing methods in accuracy and semantic alignment. This unified framework integrates vision and semantics, providing biologically interpretable insights into brain-inspired semantic cognition.

# Repository Contents

- [config](./config): Configuration files directory.
- [data](./data): We have provided some of the data on GitHub. For the complete data set, please visit [here](https://huggingface.co/datasets/AthensJ/DecodingSemantics/tree/main).
- [README.md](./README.md): Project documentation.
- [model.py](./model.py): Model structure definitions.
- [test.py](./test.py): Testing script.
- [train.py](./train.py): Training script.


# Environmental Requirements

## Hardware Requirements

Our model is deployed on a server with 4 cores, 32g memory, and dual RTX4090 GPUs.

## Software Requirements

Linux: Ubuntu 22.04  


# Installation Guide
Run ``setup.sh`` to quickly create a conda environment that contains the packages necessary to run our scripts; activate the environment with conda activate our work.


```
. setup.sh
```

Additional environments needed to run all the code:
```
pip install open_clip_torch

pip install transformers==4.28.0.dev0
pip install diffusers==0.24.0
```


# Demo

## Visual-Semantic Retrieval
We provide a script for training on the dataset we collected. Please modify the dataset path and run:
```
cd DeocdingSemantic/
python retrieval_train.py --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```

Additionally, replicating the results of other subjects:
```
cd DeocdingSemantic/
python retrieval_allsub.py --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```
We provide the script to evaluation the models visual-semantic retrieval:
```
cd DeocdingSemantic/
test.py
```

## Visual Reconstruction
We provide quick training and inference scripts for visual reconstruction. Please modify your data set path and run zero-shot on test dataset:
```
#Get visual neural embeddings aligned with clip embedding:
python train.py --visual modalities --gpu cuda:0  --output_dir ./outputs/contrast
```
```
#Reconstruct images by assigning modalities and subjects:
python test.py
```




# Results

In this [benchmark comparison](http://docs.neurodata.io/lol/lol-paper/figures/real_data.html), we show that LOL does better than all linear embedding techniques in supervised HDLSS settings when dimensionality is high (d > 100, ntrain <= d) on 20 benchmark problems from the [UCI](https://archive.ics.uci.edu/ml/index.php) and [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks) datasets. LOL provides a good tradeoff between maintaining the class conditional difference (good misclassification rate) in a small number of dimensions (low number of embedding dimensions).

# License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

# Issues

If you have any questions, please visit [issues](https://github.com/HolderXJTU/DecodingSemantic/issues).

# Citation

If you find our work useful, please consider citing:


```
@article{Ma2025Decoding,
  title={Decoding Semantic Concept Through Brain fMRI Spatio-Temporal Voxel Patterns},
  author={Ma, Yongqiang and Jing, Haodong and Zheng, Nanning},
    journal={Nature Computational Science under review},
  year={2025}
}
```
