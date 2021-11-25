# VRDL HW2 - Street View House Numbers detection
###### tags: `基於深度學習之視覺辨識專論`
## Introduction
![](https://i.imgur.com/NTT13MA.jpg)

In this assignment, we are given SVHN dataset, which contains 33,402 trianing images, 13,068 test images. Our goal is to train a accurate & fast digit detector.

I use detectron2 Faster RCNN for this assignment. I only include the files I created/modified in this repository. Please refer to the installation part below for more information.

## Environment
### Hardware
* CPU: AMD Ryzen 5 3600 6-Core
* GPU: NVIDIA GeForce RTX 3070 8GB

### Conda environment
The used packages are listed in: [requirement.txt](https://drive.google.com/file/d/1VramN_qDKc84G0tdbePsmDLBieb-zmnz/view?usp=sharing)

Then, you can create a conda environment named `detectron2` by the command:
```
conda create --name detectron2 --file requirements.txt
```

## Installation
1. First, activate the conda environment named `detectron2`.
```
conda activate detectron2
```

2. `cd` to a desired folder, and then install detectron2.
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

3. Place `myTrain.py`, `mtTest.py`, and other files in this repository into the root folder (where you just installed detectron2).


## Prepare dataset
1. Create a folder named `myDataset` under root.
2. Use `read_digitStruct.py` to convert `digitStruct.mat` to  `digitStruct.json`.
(Or download `digitStruct.json` [here](https://drive.google.com/file/d/1Rd7IfFrB5CPz6bpvJaesH8I2tXZ_hLRT/view?usp=sharing))
3. Put SVHN train, test, and `digitStruct.json` into `myDataset` folder.

## Train & Test
### Train
To train the model, run this command:
```
python myTrain.py
```
Note: To train from scratch, download the pretrained model of Faster R-CNN from [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#faster-r-cnn). And put the pretrained model file under the root folder.

### Test (Inference)
To test the model and output `answer.json`, create a folder named `output` and put [model_final.pth](https://drive.google.com/file/d/132S-PPWmu94KEXdnRephOi8oZ2VIub-V/view?usp=sharing) inside. Then, run this command:
```
python myTest.py
```
It will generate `answer.json` file under `output` folder.

## Inference the first 100 test images on Google Colab
https://colab.research.google.com/drive/1_Yy9I9SxHyVdY44m3mSJ7RLSgsAmMP7Z?usp=sharing

## Results
The score on CodaLab and the best speed I can get on Google Colab.

| Model Name  | Score | Best Speed on Colab |
| :-: | :-: | :-: |
| [model_final.pth](https://drive.google.com/file/d/132S-PPWmu94KEXdnRephOi8oZ2VIub-V/view?usp=sharing)  |  0.39791  | 0.28844 |

![](https://i.imgur.com/WsXcQ9J.png)
