# Decoding Semantic Concept Through Brain fMRI Spatio-Temporal Voxel Patterns


![Python](https://img.shields.io/badge/python-3.8%2B-blue)


## Contents

- [Overview](#overview)
- [Repository  Contents](#repo-contents)
- [Environmental Requirements](#system-requirements)
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
- [data](./data): We have provided some of the data on GitHub. For the complete data set, please visit here.
- [README.md](./README.md): Project documentation.
- [model.py](./model.py): Model structure definitions.
- [test.py](./test.py): Testing script.
- [train.py](./train.py): Training script.


# Environmental Requirements

## Hardware Requirements

Our model is deployed on a server with 4 cores, 32g memory, and dual RTX4090 cards.

## Software Requirements

Linux: Ubuntu 22.04  


# Installation Guide

## Stable Release

`lolR` is available in a stable release on CRAN:

```
install.packages('lolR')
```

## Development Version

### Package dependencies

Users should install the following packages prior to installing `lolR`, from an `R` terminal:

```
install.packages(c('ggplot2', 'abind', 'irlba', 'knitr', 'rmarkdown', 'latex2exp', 'MASS', 'randomForest'))
```

which will install in about 30 seconds on a machine with the recommended specs.

The `lolR` package functions with all packages in their latest versions as they appear on `CRAN` on December 13, 2017. Users can check [CRAN snapshot](https://mran.microsoft.com/timemachine/) for details. The versions of software are, specifically:
```
abind_1.4-5
latex2exp_0.4.0
ggplot2_2.2.1
irlba_2.3.1
Matrix_1.2-3
MASS_7.3-47
randomForest_4.6-12
```

If you are having an issue that you believe to be tied to software versioning issues, please drop us an [Issue](https://github.com/neurodata/lol/issues). 

### Package Installation

From an `R` session, type:

```
require(devtools)
install_github('neurodata/lol', build_vignettes=TRUE, force=TRUE)  # install lol with the vignettes
require(lolR)
vignette("lol", package="lolR")  # view one of the basic vignettes
```

The package should take approximately 40 seconds to install with vignettes on a recommended computer. 

# Demo

## Functions

For interactive demos of the functions, please check out the vignettes built into the package. They can be accessed as follows:

```
require(lolR)
vignette('lol')
vignette('pca')
vignette('cpca')
vignette('lrcca')
vignette('mdp')
vignette('xval')
vignette('qoq')
vignette('simulations')
vignette('nearestCentroid')
```

## Extending the lolR Package

The lolR package makes many useful resources available (such as embedding and cross-validation) for simple extension. 

To extend the lolR package, check out the vignettes:

```
require(lolR)
vignette('extend_embedding')
vignette('extend_classification')
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
