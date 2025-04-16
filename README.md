# Cityscapes Semantic Segmentation with UNet
This project implements a UNet-based semantic segmentation model trained on the **Cityscapes dataset** using PyTorch.

## Dataset: Cityscapes (from Kaggle)

We use two parts of the Cityscapes dataset:

- `leftImg8bit`: Raw images
- `gtFine`: Ground truth segmentation masks

### How to Download the Dataset

1. Go to [Kaggle Datasets](https://www.kaggle.com/) and search for:
   - **Cityscapes Images**: [`cityscapes-leftimg8bit-trainvaltest`](https://www.kaggle.com/datasets/dansbecker/cityscapes-leftimg8bit-trainvaltest)
   - **Cityscapes Ground Truth**: [`gtfine-trainvaltest`](https://www.kaggle.com/datasets/dansbecker/gtfine-trainvaltest)

2. Download the ZIP files and extract them into the project directory like this:
your_project/ ├── leftImg8bit/ │ └── train/ val/ test/ ├── gtFine/ │ └── train/ val/ test/

