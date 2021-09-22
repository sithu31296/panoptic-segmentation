# <div align="center">Panoptic Segmentation (WIP)</div>

<div align="center">
<p>Easy to use SOTA Panoptic Segmentation models in PyTorch</p>
</div>

## <div align="center">Model Zoo</div>

[pvtv2]: https://arxiv.org/abs/2106.13797
[resnet]: https://arxiv.org/abs/1512.03385

[panopticsegformer]: https://arxiv.org/abs/2109.03814
[maskformer]: http://arxiv.org/abs/2107.06278
[panopticdeeplab]: https://arxiv.org/abs/1911.10194

* Supported Backbones: [ResNet][resnet], [PVTv2][pvtv2]
* Supported Models: [PanopticDeepLab][panopticdeeplab]


<details open>
  <summary><strong>COCO-val</strong></summary>

Model | Backbone | PQ | PQ<sup>th | PQ<sup>st | Params <br><sup>(M) | GFLOPs | Weights
--- | --- | --- | --- | --- | --- | --- | ---
[Panoptic SegFormer][panopticsegformer] | PVTv2-B0 | 49.6 | 55.5 | 40.6 | 22 | 156 | -
|| PVTv2-B2 | 52.6 | 58.7 | 43.3 | 42 | 219 | -
|| PVTv2-B5 | 54.1 | 60.4 | 44.6 | 101 | 391 | -
[MaskFormer][maskformer] | Swin-T | 47.7 | 51.7 | 41.7 | 42 | 179 | -
|| Swin-S | 49.7 | 54.4 | 42.6 | 63 | 259 | -
|| Swin-B | 51.1 | 56.3 | 43.2 | 102 | 411 | -

</details>

<details>
  <summary><strong>COCO-test</strong> (click to expand)</summary>

Model | Backbone | PQ | PQ<sup>th | PQ<sup>st | Params <br><sup>(M) | GFLOPs | Weights
--- | --- | --- | --- | --- | --- | --- | ---
[Panoptic SegFormer][panopticsegformer] | PVTv2-B5 | 54.4 | 61.1 | 44.3 | 101 | 391 | -

</details>

## <div align="center">Supported Datasets</div>

[coco]: https://cocodataset.org/#home
[cityscapes]: https://www.cityscapes-dataset.com/

Dataset | Type | Categories | Train <br><sup>Images | Val<br><sup>Images | Test<br><sup>Images | Image Size<br><sup>(HxW)
--- | --- | --- | --- | --- | --- | ---
[COCO][coco] | General | 171 | 118,000 | 5,000 | 20,000 | -
[CityScapes][cityscapes] | Street | 19 | 2,975 | 500 | 1,525<sup>+labels | 1024x2048

<details>
  <summary><strong>Datasets Structure</strong> (click to expand)</summary>

Datasets should have the following structure:

```
data
|__ CityScapes
    |__ leftImg8bit
        |__ train
        |__ val
        |__ test
    |__ gtFine
        |__ train
        |__ val
        |__ test

|__ COCO
    |__ images
        |__ train2017
        |__ val2017
    |__ labels
        |__ train2017
        |__ val2017
```
</details>

## <div align="center">Usage (Coming Soon)</div>

<details>
  <summary><strong>Requirements</strong> (click to expand)</summary>

* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

</details>

<br>
<details>
  <summary><strong>Configuration</strong> (click to expand)</summary>

Create a configuration file in `configs`. Sample configuration for Custom dataset can be found [here](configs/custom.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

To train with a single GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE.yaml
```

To train with multiple gpus, set `DDP` field in config file to `true` and run as follows:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE.yaml
```

</details>

<br>
<details>
  <summary><strong>Evaluation</strong> (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE.yaml
```

To evaluate with multi-scale and flip, change `ENABLE` field in `MSF` to `true` and run the same command as above.

</details>

<br>
<details open>
  <summary><strong>Inference</strong></summary>

Make sure to set `MODEL_PATH` of the configuration file to model's weights.

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE.yaml
```

</details>

<details>
  <summary><strong>References</strong> (click to expand)</summary>


</details>

<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

```
@misc{li2021panoptic,
  title={Panoptic SegFormer}, 
  author={Zhiqi Li and Wenhai Wang and Enze Xie and Zhiding Yu and Anima Anandkumar and Jose M. Alvarez and Tong Lu and Ping Luo},
  year={2021},
  eprint={2109.03814},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{cheng2020panopticdeeplab,
  title={Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation}, 
  author={Bowen Cheng and Maxwell D. Collins and Yukun Zhu and Ting Liu and Thomas S. Huang and Hartwig Adam and Liang-Chieh Chen},
  year={2020},
  eprint={1911.10194},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

</details>