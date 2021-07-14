<div align="center">

# Fast and Accurate Lane Detection via Frequency Domain Learning 
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2010.12035-b31b1b.svg)](https://arxiv.org/abs/2010.12035)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/keep-your-eyes-on-the-lane-attention-guided/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=keep-your-eyes-on-the-lane-attention-guided) -->
</div>

This repository holds the source code of the paper: "_Fast and Accurate Lane Detection via Frequency Domain Learning_" (Link:). We introduce multi-frequency analysis into lane detection to achieve high accuracy while without much speed delay. (**This repository is heavily based on the state-of-the-art lane detection model [msld](https://arxiv.org/abs/2010.12035).**)

**Our paper has been accepted to ACMMM'21.** 

![Method overview](data/figures/architecture.jpg "Method overview")

![MSFC and MSFA module](data/figures/MSFC+MSFA.jpg "MSFC and MSFA overview")

### Table of contents
- [Fast and Accurate Lane Detection via Frequency Domain Learning](#fast-and-accurate-lane-detection-via-frequency-domain-learning)
    - [Table of contents](#table-of-contents)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Install](#2-install)
    - [3. Getting started](#3-getting-started)
      - [Datasets](#datasets)
      - [Training & testing](#training--testing)
      - [Reproducing a result from the paper](#reproducing-a-result-from-the-paper)


### 1. Prerequisites
- Python >= 3.5
- PyTorch == 1.6, tested on CUDA 10.2. The models were trained and evaluated on PyTorch 1.6. When testing with other versions, the results (metrics) are slightly different.
- CUDA, to compile the NMS code
- Other dependencies described in `requirements.txt`

The versions described here were the lowest the code was tested with. Therefore, it may also work in other earlier versions, but it is not guaranteed (e.g., the code might run, but with different outputs).

### 2. Install
Conda is not necessary for the installation, as you can see, I only use it for PyTorch and Torchvision.
Nevertheless, the installation process here is described using it.

```bash
conda create -n msld python=3.8 -y
conda activate msld
conda install pytorch==1.6 torchvision -c pytorch
pip install -r requirements.txt
cd lib/nms; python setup.py install; cd -
```

### 3. Getting started
#### Datasets
For a guide on how to download and setup each dataset, see [DATASETS.md](DATASETS.md).

#### Training & testing
Train a model:
```
python main.py train --exp_name example --cfg example.yml
```
For example, to train msld with the ResNet-34 backbone on TuSimple, run:
```
python main.py train --exp_name msld_r34_tusimple --cfg cfgs/msld_tusimple_resnet34.yml
```
After running this command, a directory `experiments` should be created (if it does not already exists). Another
directory `msld_r34_tusimple` will be inside it, containing data related to that experiment (e.g., model checkpoints, logs, evaluation results, etc)

Evaluate a model:
```
python main.py test --exp_name example
```
This command will evaluate the model saved in the last checkpoint of the experiment `example` (inside `experiments`).
If you want to evaluate another checkpoint, the `--epoch` flag can be used. For other flags, please see `python main.py -h`. To **visualize the predictions**, run the above command with the additional flag `--view all`.

#### Reproducing a result from the paper
0. Set up the dataset you want to reproduce the results on (as described in [DATASETS.md](DATASETS.md)).
1. Download the zip containing all pretrained models  and then unzip it at the code's root:
```
Baidu Cloud Disk "https://pan.baidu.com/s/1IosBoyd26zBTUEcg4s5Gsw". Extraction Code: "g2be". 
unzip experiments.zip
```
<!-- ```bash
gdown "https://drive.google.com/uc?id=1R638ou1AMncTCRvrkQY6I-11CPwZy23T" # main experiments on TuSimple, CULane and LLAMAS (1.3 GB)
or Baidu Cloud Disk "https://pan.baidu.com/s/1IosBoyd26zBTUEcg4s5Gsw". Extraction Code: "g2be". 
unzip msld_experiments.zip
``` -->
1. Run the evaluation (inference + metric computation):
```bash
python main.py test --exp_name $EXP_NAME
```
Replace `$EXP_NAME` with the name of a directory inside `experiments/`. For instance, if you want to reproduce the results using the ResNet-34 backbone on the TuSimple dataset, run:
```bash
python main.py test --exp_name msld_r34_tusimple
```

### 4. Results
![F1 vs. Latency for state-of-the-art methods on lane detection](data/figures/acc.jpg "F1 vs. Latency for state-of-the-art methods on lane detection")

#### CULane

|   Backbone    |        F1 (%)      | FPS | MACs(G) |
|     :---      |         ---:                       |   ---:            | ---:|
| ResNet-18     | 76.32                              |  237            | 9.4 |
| ResNet-34     | 76.98                              |  164            | 18.1 |


#### TuSimple
|   Backbone    |      Accuracy (%)     |      FDR (%)     |      FNR (%)     |      F1 (%)     | FPS |
|    :---       |         ---:          |       ---:       |       ---:       |      ---:       | ---:|
| ResNet-18     |    95.62             |    3.37         |    3.07          |    96.77        | 237 |
| ResNet-34     |    95.73              |    3.26          |    3.05          |    96.84        | 164 |


#### LLAMAS
|   Backbone    |      F1 (%)     |   Precision (%)  |   Recall (%)  | FPS |
|    :---       |         ---:    |       ---:       |       ---:    | ---:|
| ResNet-18     |      93.73      |     96.77        |    90.88      | 237 |
| ResNet-34     |      93.77      |     96.88        |    90.85      | 164 |

**Additional results can be seen in the paper.**

### 5. Code structure
**This repository is based on the state-of-the-art lane detection model [msld](https://arxiv.org/abs/2010.12035).**


- **cfgs:** Default configuration files
- **figures:** Images used in this repository
- **lib**
  - **datasets**
    - **culane.py:** CULane annotation loader
    - **lane_dataset.py:** Transforms raw annotations from a `LaneDatasetLoader` into a format usable by the model
    - **lane_dataset_loader.py:** Abstract class that each dataset loader implements
    - **llamas.py:** LLAMAS annotation loader
    - **nolabel_dataset.py:** Used on data with no annotation available (or quick qualitative testing)
    - **tusimple.py:** TuSimple annotation loader
   - **models:**
     - **layer.py** MSFC and MSFA implementation
     - **msld.py:** MSLD implementation
     - **matching.py:** Utility function for ground-truth and proposals matching
     - **resnet.py:** Implementation of ResNet
  - **nms:** NMS implementation
  - **config.py:** Configuration loader
  - **experiment.py:** Tracks and stores information about each experiment
  - **focal_loss.py:** Implementation of Focal Loss
  - **lane.py:** Lane representation
  - **runner.py:** Training and testing loops
- **utils**:
  - **culane_metric.py:** Unofficial implementation of the CULane metric. This implementation is faster than the oficial,
  however, it does not matches exactly the results of the official one (error in the order of 1e-4). Thus, it was used only during the model's development.
  For the results reported in the paper, the official one was used.
  - **gen_anchor_mask.py**: Computes the frequency of each anchor in a dataset to be used in the anchor filtering step
  - **gen_video.py:** Generates a video from a model's predictions
  - **llamas_metric.py**: Official implementation of the LLAMAS metric
  - **llamas_utils.py**: Utilities functions for the LLAMAS dataset
  - **speed.py:** Measure efficiency-related metrics of a model
  - **tusimple_metric.py**: Official implementation of the TuSimple metric
  - **viz_dataset.py**: Show images sampled from a dataset (post-augmentation)
- **main.py:** Runs the training or testing phase of an experiment


<!-- ### 6. Citation
If you use this code in your research, please cite:

```bibtex
@misc{msld2020arxiv,
  author    = {Lucas Tabelini
               and Rodrigo Berriel
               and Thiago M. Paix\~ao
               and Claudine Badue
               and Alberto Ferreira De Souza
               and Thiago Oliveira-Santos},
  title     = {{Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection}},
  eprint    = {arXiv:2010.12035},
  year      = {2020}
}
``` -->
