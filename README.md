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

## Task 1

### To convert the Cityscapes ground truth annotations into segmentation masks suitable for training a semantic segmentation model. These masks must be:

1. In the correct shape and format
2. Mapped to valid class indices
3. Consistent across training, validation, and test splits
4. Class Mapping Strategy
The original Cityscapes dataset has 34 classes, but not all are useful or relevant for training. You mapped only the 19 valid classes using a dictionary and unwanted classes were ignored by setting them to 255.

### Handling edge cases and overlapping regions
1.The function convert_mask transforms Cityscapes labelIds masks into multi-class segmentation masks using a predefined mapping of valid class labels.
2. A new mask is initialized with value 255 to mark ignore regions.
3. Pixels are remapped from original labels to new class indices, and unexpected labels remain as 255.
4. Overlapping masks aren’t a concern in Cityscapes, so direct pixel replacement suffices.
5. Corrupted annotation files are caught with exceptions and logged without interrupting processing.
6. process_subset iterates through each city and image, ensuring structure compatibility with Cityscapes.
7. Each image is paired with its corresponding ground truth annotation using a consistent naming pattern.
8. Missing annotation files are skipped with a warning to maintain robustness.
9. The processing stops once a specified max_images limit is reached for reproducibility and efficiency.
10. All generated masks are saved to the output directory with _mask.png suffix for training segmentation models.

## Task 2
Model Architecture: UNet (Why UNet?)
We chose the UNet architecture due to the following reasons:
Encoder-Decoder Design: It captures both high-level context (encoder) and precise localization (decoder), which is crucial for pixel-wise tasks like semantic segmentation.
Skip Connections: Helps retain spatial information lost during downsampling by connecting encoder and decoder layers at the same resolution.
Lightweight Yet Powerful: UNet is relatively lightweight compared to more modern architectures like DeepLabV3 or HRNet, making it suitable for quick experimentation or limited compute.
Proven Effectiveness: It has demonstrated great performance on segmentation benchmarks like medical imaging and general-purpose segmentation tasks.

Implementation Details
Framework used -	PyTorch
Model - UNet
Loss Function - CrossEntropyLoss (with ignore_index=255 for unlabeled pixels)
Optimizer - Adam
Learning Rate - 1e-4
Batch Size - 8(modifiable based on GPU memory)
Epochs - 10 (For reducing the training time )
Evaluation Metrics - Mean IoU, Per-Class IoU, Pixel Accuracy
Logger - Weights & Biases (wandb)

### Loss-function 
CrossEntropyLoss is the standard choice because it compares the predicted class probabilities at each pixel with the true class label.

Cityscapes masks contain pixels labeled 255 which represent unlabeled or irrelevant regions.

### Optimizer 
Adam is an adaptive optimizer that combines the benefits of AdaGrad (good for sparse data) and RMSProp (good for non-stationary objectives).It adjusts the learning rate for each parameter individually and works well out-of-the-box for deep networks like UNet.

### Learning rate
Why 1e-4?
This is a safe default for Adam that prevents overshooting while still allowing decent convergence.

### Metrics: Mean IoU, Pixel Accuracy
Why Mean IoU?
It's a widely-used metric for segmentation tasks that reflects how well each class is segmented.
Averaging IoU over all classes ensures performance isn't biased toward frequent classes (like "road").

Why Pixel Accuracy?
Gives an intuitive measure of how many pixels are classified correctly overall.
Useful but can be misleading if the dataset is imbalanced — which is why it's reported alongside mean IoU.

# Results 
epochs used	 - 10
mean_iou	achieved  - 0.24408
pixel_accuracy	 -  0.86996
train_loss  - 	0.59682
val_loss  -  0.46705

# Experiments
Model: UNet
Chosen for its encoder-decoder structure with skip connections, making it effective for precise pixel-wise segmentation on complex datasets like Cityscapes.
Loss Function: CrossEntropyLoss(ignore_index=255)

Suitable for multi-class pixel classification.
ignore_index=255 excludes unlabeled/irrelevant regions from loss computation, aligning with Cityscapes' annotation format.

Optimizer: Adam
Adaptive optimizer well-suited for deep learning tasks, with good convergence and stability.

Learning Rate: 1e-4
A balanced value for Adam that avoids overshooting while ensuring steady progress.

Batch Size: 8
Kept small due to high-resolution images and GPU memory constraints.

Epochs: 10
Chosen based on convergence trends; early stopping or model checkpointing can avoid overfitting.

Metrics Used:
Mean IoU: Captures class-wise intersection-over-union, a robust segmentation metric.
Pixel Accuracy: Simple but effective for measuring overall correctness (less informative with class imbalance).



















