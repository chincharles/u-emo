<div align="center">
    <img src="./figure/logo.png" height="150px" />
</div>



By [Chuang Chen](https://github.com/chincharles), [Xiao Sun](https://orcid.org/0000-0001-9750-7032)\*, [Zhi Liu](https://orcid.org/0000-0003-0537-4522).

This repo is the official implementation of ["UniEmoX: Cross-modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception"](https://ieeexplore.ieee.org/document/11080231).

<p align="center">
  <a href='https://arxiv.org/abs/2409.18877'>
  <img src='https://img.shields.io/badge/Arxiv-2409.18877-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a> 
  <a href='https://arxiv.org/abs/2409.18877'>
  <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a> 
  <a href='https://ieeexplore.ieee.org/document/11080231/media#media'>
  <img src='https://img.shields.io/badge/Appendix-IEEE--Xplore-blue?style=flat&logo=ieee&logoColor=white'>
  </a>
</p>

## 📰 News

📅***09/30/2024***

- Pre-trained and fine-tuned models on diverse datasets have been released.
- Split scripts and processed datasets for six benchmarks (including Emo8) have been provided.

📅***07/02/2025***

- 🎉 The paper has been officially accepted by **IEEE Transactions on Image Processing (TIP)**.

📅***07/17/2025***

- Training and fine-tuning scripts have been released.

## 🔎 Abstract

Visual emotion analysis holds significant research value in both computer vision and psychology. However, existing methods for visual emotion analysis suffer from limited generalizability due to the ambiguity of emotion perception and the diversity of data scenarios. To tackle this issue, we introduce UniEmoX, a cross-modal semantic-guided large-scale pretraining framework. Inspired by psychological research emphasizing the inseparability of the emotional exploration process from the interaction between individuals and their environment, UniEmoX integrates scene-centric and person-centric low-level image spatial structural information, aiming to derive more nuanced and discriminative emotional representations. By exploiting the similarity between paired and unpaired image-text samples, UniEmoX distills rich semantic knowledge from the CLIP model to enhance emotional embedding representations more effectively. To the best of our knowledge, this is the first large-scale pretraining framework that integrates psychological theories with contemporary contrastive learning and masked image modeling techniques for emotion analysis across diverse scenarios. Additionally, we develop a visual emotional dataset titled Emo8. Emo8 samples cover a range of domains, including cartoon, natural, realistic, science fiction and advertising cover styles, covering nearly all common emotional scenes. Comprehensive experiments conducted on seven benchmark datasets across two downstream tasks validate the effectiveness of UniEmoX.

## 🔧 Technical Solution

<div align="center">
    <img src="./figure/structure.png" height="360px" />
</div>


## ⚙️ Getting Started

### Installation

- Install `CUDA 11.3` with `cuDNN 8` following the official installation guide of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).
- Setup conda environment:

```bash
# Create environment
conda create -n uniemo python=3.8 -y
conda activate uniemox

# Install requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# Clone UniEmoX
git clone https://github.com/chincharles/u-emo.git
cd u-emo

# Install other requirements
pip install -r requirements.txt
```

### Preparing Datasets

To ensure the reproducibility of results, you can use our data splits for training and testing.

|                             UBE                              |                              FI                              |                             Emo8                             |                            CAER-S                            |                             HECO                             |                           Emotion6                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [download](https://pan.baidu.com/s/1XCf3yOTEQwJX3G50rwTiIA?pwd=989c) | [download](https://pan.baidu.com/s/1dw01Jd_pKBNotE7AGl8PLA?pwd=7mea) | [download](https://drive.google.com/drive/folders/1bE9FfqQCSo8fKaoy-kOvanwLVMvPMZ8d?usp=sharing) | [download](https://pan.baidu.com/s/11jbAp2cpnM53Xu8w2hkePw?pwd=bzjx) | [download](https://pan.baidu.com/s/19aV6TwnAapiP6_LPXsxR_g?pwd=ru8d) | [download](https://pan.baidu.com/s/1eS1WQiv2Ozx-E4Ef1ibaRA?pwd=8hyr) |

You can also refer to `<root-path>/u-emo/datasets/datasets.py` to reproduce the process of splitting the original data into our dataset. Please download the pre-trained dataset **EmoSet** from the [link](https://github.com/JingyuanYY/EmoSet).

### Pre-trained Models

The following table provides pre-trained models using `UniEmoX` with 100 and 200 epochs.

| Pre-train Epochs |                      Pre-trained Model                       |
| :--------------: | :----------------------------------------------------------: |
|       100        | [download](https://pan.baidu.com/s/1RRJMZ-WacKeLOLv_kj3G5Q?pwd=q47b) |
|       200        | [download](https://pan.baidu.com/s/1sN97Q5vJ56GOXo6KtpfqSw?pwd=fn62) |

### Pre-training with UniEmoX

To pre-train models with `UniEmoX`, run:

```bash
CUDA_VISIBLE_DEVICES=<gpus-to-use> python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_pretrain.py \ 
--data_path <emoset-path> --output_dir <output-directory> --resume <>
```

Please note that the parameters `data_path` and `output_dir` must be provided according to your location. You can also adjust other parameters in `main_pretrain.py` according to your needs. Please download the initialization model of MAE from the [link](https://pan.baidu.com/s/1o8nKD_F8KOIPVb0Rsn6ioQ?pwd=pxh6), and load the pre-trained weights using the `--resume` parameter. The YOLOv3 components required for the pretraining process can be downloaded from the [link](https://pan.baidu.com/s/15GVMvdBamfglqSal-QlUcw?pwd=5bux). The CLIP component required for the pretraining process can be downloaded from the [link](https://pan.baidu.com/s/1GKjSEPIT9JyTkJzK6W1oPg?pwd=9175).

### Fine-tuned Models

The following table provides models fine-tuned for 100 epochs on our proposed Emo8 dataset and five other benchmark datasets using pre-trained models.

|                             UBE                              |                              FI                              |                             Emo8                             |                            CAER-S                            |                             HECO                             |                           Emotion6                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [download](https://pan.baidu.com/s/1S3vy7UHEMv_8UEbdIiL6qQ?pwd=bfa8) | [download](https://pan.baidu.com/s/142gnWOVzHGC_kQufIQBR4A?pwd=t4i6) | [download](https://pan.baidu.com/s/17xDnqDsWHByAodUe0WWqTQ?pwd=dywi) | [download](https://pan.baidu.com/s/1V3uUvYlM2PAqwPcHK4w0bA?pwd=mx7r) | [download](https://pan.baidu.com/s/1ZGmD_g3BUiiSLm5jYaQjJQ?pwd=wvge) | [download](https://pan.baidu.com/s/13CONfmdytokTJd9S5UFNOw?pwd=t7yz) |

### Fine-tuning pre-trained models
To fine-tune models pre-trained by `UniEmoX`, run:
```bash
CUDA_VISIBLE_DEVICES=<gpus-to-use> python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_finetune.py \ 
--finetune <pretrained-ckpt> --data_path <datasets-base-path> --output_dir <output-directory> --dataset <dataset-to-use>
```

Please note that the parameters `finetune`, `data_path`, `output_dir` and `dataset` must be provided according to your location. You can also adjust other parameters in `main_finetune.py` according to your needs.

### Evaluating fine-tuned models

To evaluate the fine-tuned models on our proposed Emo8 dataset and five other public benchmark datasets, run: 

```bash
CUDA_VISIBLE_DEVICES=<gpus-to-use> python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_finetune.py \
--eval --resume <fine-tuned checkpoint> --data-path <datasets-base-path> --dataset <dataset-to-use>
```

Please note that the parameters `eval`, `resume`, `data-path` and `dataset` must be provided according to your location. You can also adjust other parameters in `main_finetune.py` according to your needs.


## ❤️ Acknowledgements

This repo is based on [MAE](https://github.com/facebookresearch/mae), [YOLOv3](https://arxiv.org/abs/1804.02767) and [CLIP](https://github.com/openai/CLIP).

## 📧 Contact Information

For help or issues using `UniEmoX`, please submit a GitHub issue.

For other communications, please contact `eric.chuangchen@gmail.com`.

## 🎓Citing UniEmoX

If you find this work helpful, please cite our paper:

**IEEE Published Version**

```latex
@ARTICLE{11080231,
  author={Chen, Chuang and Sun, Xiao and Liu, Zhi},
  journal={IEEE Transactions on Image Processing}, 
  title={UniEmoX: Cross-modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2025.3587577}
}
```

**Preprint Version**

```latex
@article{UniEmoX2024,
  title={UniEmoX: Cross-modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception}, 
  author={Chuang Chen and Xiao Sun and Zhi Liu},
  journal={arXiv preprint arXiv:2409.18877},
  year={2024}
}
```

## 📚 License

This code is distributed under an [MIT LICENSE](). Note that our code depends on other libraries and datasets which each have their own respective licenses that must also be followed.
