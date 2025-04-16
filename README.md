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
## Model Architecture

I implemented a **U-Net** architecture for this segmentation task. U-Net was chosen for its:

1. Encoder-decoder structure with skip connections that preserves spatial information
2. Ability to segment fine details in images
3. Relatively lightweight architecture that can be trained with limited computational resources
4. Strong performance on medical and satellite image segmentation tasks, which share similarities with urban scene segmentation

### Model Details

- **Encoder**: 5 downsampling blocks with double convolutions and max pooling
- **Decoder**: 5 upsampling blocks with transposed convolutions and skip connections
- **Input size**: 512×512×3 (RGB images)
- **Output size**: 512×512×19 (probability maps for each class)
- **Parameters**: ~31 million
- **Activation functions**: ReLU in hidden layers, Softmax in the output layer

## Loss Function

I used a combination of **Dice Loss** and **Cross Entropy Loss** for training:

- **Cross Entropy Loss**: Good for classification tasks, focuses on per-pixel accuracy
- **Dice Loss**: Directly optimizes the IoU metric, better handles class imbalance
- **Combined Loss**: Weighted sum of both losses (0.5 * CE + 0.5 * Dice) to get the benefits of both

This combined loss function helped address the class imbalance problem in the dataset, where some classes like 'road' and 'building' are much more common than others like 'traffic light' or 'bicycle'.

## Training Details

- **Optimizer**: Adam with learning rate 1e-4
- **Learning rate scheduler**: ReduceLROnPlateau to reduce learning rate when validation loss plateaus
- **Batch size**: 8
- **Epochs**: 30
- **Early stopping**: Based on validation loss with patience of 5 epochs
- **Hardware**: Trained on NVIDIA GPU with 16GB VRAM
- **Training time**: Approximately 8 hours

## Results

The model achieved the following metrics on the test set:

- **Mean IoU**: 0.24408
- **Pixel Accuracy**: 0.86996
- **Mean Dice**: 0.3392

  
### Visualization

![Example Predictions](path/to/example_predictions.png)

The training is also visualized on wandb and can be accessed with following links.
View run unet-experiment at: https://wandb.ai/ishanagar205-indian-institute-of-science/cityscapes-segmentation/runs/kmkwhpyf
View project at: https://wandb.ai/ishanagar205-indian-institute-of-science/cityscapes-segmentation
Synced 5 W&B file(s), 18 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20250416_091242-kmkwhpyf/logs

## Future Improvements

1. **Architecture**: Experiment with more advanced architectures like DeepLabV3+ or HRNet
2. **Backbone**: Use pre-trained backbones like ResNet or EfficientNet for the encoder
3. **Loss function**: Implement Focal Loss to further address class imbalance
4. **Post-processing**: Add CRF (Conditional Random Fields) for boundary refinement
5. **Ensemble**: Combine predictions from multiple models for better performance

## References

1. Cityscapes Dataset: [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
2. U-Net Paper: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)




