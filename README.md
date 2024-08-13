# Enhanced Multi-modal Transformer with Mamba Decoding for Precise Video Moment Retrieval

<p align="center"><img width="850" src="https://raw.githubusercontent.com/TencentARC/UMT/main/.github/model.svg"></p>

## Installation

Please see below for the main environment settings we use. If you encounter any problems during the automatic installation, you can install these packages yourself.

- CUDA 11.8.0
- PyTorch 2.1.1+cu118
- torchtext 0.16.1
- torchvision 0.16.1+cu118
- causal-conv1d      1.1.0
- mamba-ssm 1.1.1
- [NNCore](https://github.com/yeliudev/nncore) 0.3.6


## Getting Started

### Download and prepare the datasets

1. Download and unzip the Charades-STA feature into datasets.

   https://drive.google.com/file/d/1nI4shVD0PYsawOaOd-nNyX6ukzrIrxO3/view?usp=drive_link

2. Prepare the files in the following structure.

```
UMT
├── configs
├── datasets
├── models
├── tools
├── data
│   ├── charades
│   │   ├── *features
│   │   └── charades_sta_{train,test}.txt
├── README.md
├── setup.cfg
└── ···
```

### Train a model

Run the following command to train a model using a specified config.

```shell
# Single GPU
python tools/launch.py ${path-to-config}

```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
python tools/launch.py ${path-to-config} --checkpoint ${path-to-checkpoint} --eval
```


