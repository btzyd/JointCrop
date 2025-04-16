# Enhancing Contrastive Learning Inspired by the Philosophy of “the Blind Men and the Elephant”

This is a PyTorch implementation of the [JointCrop paper (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34425). The code is mainly modified from [MoCo v3](https://github.com/facebookresearch/moco-v3).

## Appendix of JointCrop

Due to AAAI's page limit, You can refer to the [arxiv version](https://arxiv.org/abs/2412.16522) for the appendix.

## Preparing the environment, code and dataset

1. Prepare the environment.

Creating a python environment and activate it via the following command.

```bash
conda create -n jointcrop python=3.9
conda activate jointcrop
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorboard timm==0.4.9 numpy==1.26.4
```

2. Clone this repository.

```bash
git clone https://github.com/btzyd/JointCrop.git
```

3. Prepare the dataset ImageNet.

Download and organize the ImageNet-1K dataset according to the same form as [MoCo v3](https://github.com/facebookresearch/moco-v3).

## Training script of JointCrop

MoCo v3 based on ResNet50 requires 2 nodes with a total of 16 GPUs and about 29GB of memory per GPU.

On the first node, run:

```python
python main_jointcrop.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  --ckpt_path './checkpoint_resnet50_ep100_jointcrop0' \
  [your imagenet-folder with train and val folders]
```

On the second node, run the same command with --rank 1.

For scripts with 300 epochs, you can refer to the [MoCo v3 repository](https://github.com/facebookresearch/moco-v3).

## Fine-tuning script of JointCrop

The fine-tuning of MoCo v3 based on ResNet50 nearly requires a total of 8 GPUs on 1 node, with about 6GB memory per GPU.

```python
python main_lincls.py \
  --dist-url 'tcp://127.0.0.1:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained 'checkpoint_resnet50_ep100_jointcrop0/checkpoint_0099.pth.tar' \
  [your imagenet-folder with train and val folders]
```

## Results of JointCrop

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">epoch</th>
<th valign="center">MoCo v3<br/>baseline</th>
<th valign="center">MoCo v3 + <br/>JointCrop(0)</th>
<!-- TABLE BODY -->
<tr>
<td align="left">100</td>
<td align="right">68.9</td>
<td align="center">69.47</td>
</tr>
<tr>
<td align="left">300</td>
<td align="right">72.8</td>
<td align="center">73.23</td>
</tr>
</tbody></table>

## Citation
```
@inproceedings{zhang2025enhancing,
  title={Enhancing Contrastive Learning Inspired by the Philosophy of “The Blind Men and the Elephant”},
  author={Zhang, Yudong and Xie, Ruobing and Chen, Jiansheng and Sun, Xingwu and Kang, Zhanhui and Wang, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22659--22667},
  year={2025}
}
```