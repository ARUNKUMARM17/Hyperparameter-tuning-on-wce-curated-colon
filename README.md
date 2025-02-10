# Image Classification Using ResNet50

## Overview
This project implements an image classification model using the **ResNet50** architecture with transfer learning and hyperparameter tuning. The model is optimized using **Keras Tuner** to find the best hyperparameters for improved performance.

## Features
- Uses **ResNet50** as the base model (pre-trained on ImageNet).
- Implements **hyperparameter tuning** with Keras Tuner.
- Applies **data augmentation** techniques to improve generalization.
- Uses **categorical crossentropy** loss for multi-class classification.
- Incorporates **early stopping** for better training efficiency.

## Dataset
The dataset consists of images stored in the following directories:
```
/content/train  # Training dataset (contains subdirectories for each class)
/content/val    # Validation dataset (contains subdirectories for each class)
```
Each subdirectory inside `train` and `val` corresponds to a different class.

## Requirements
Install the required dependencies before running the script:
```bash
pip install tensorflow matplotlib keras-tuner
```

## Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet, `include_top=False`)
- **Additional Layers**:
  - Global Average Pooling (GAP)
  - Fully connected (Dense) layers with ReLU activation
  - Batch Normalization
  - Dropout layers for regularization
  - Final Dense layer with softmax activation for classification
- **Hyperparameter Tuning**:
  - Number of units in dense layers
  - Dropout rates
  - Learning rate selection

## Hyperparameters Table
| Hyperparameter  | Values Tested |
|---------------|--------------|
| `units_1`     | 128, 256, 384, 512 |
| `dropout_1`   | 0.3, 0.4, 0.5, 0.6, 0.7 |
| `units_2`     | 64, 128, 192, 256, 320, 384, 448, 512 |
| `dropout_2`   | 0.3, 0.4, 0.5, 0.6, 0.7 |
| `learning_rate` | 1e-3, 1e-4, 1e-5 |

## Training Pipeline
1. **Data Preprocessing**:
   - Rescales pixel values to `[0,1]`.
   - Applies data augmentation: rotation, width/height shifts, shear, zoom, horizontal flips, and brightness adjustment.
2. **Model Compilation**:
   - Optimizer: Adam (learning rate tuned)
   - Loss function: Categorical Crossentropy
   - Metric: Accuracy
3. **Hyperparameter Tuning**:
   - Uses `RandomSearch` tuner to find the best hyperparameters.
4. **Training**:
   - Runs for up to 20 epochs with batch size 32.
5. **Evaluation & Visualization**:
   - Plots training/validation accuracy and loss.
6. **Model Saving**:
   - Saves the final trained model as `wcecolonfine_tuned.h5`.

## Usage
To train the model, simply run:
```bash
python train.py  # Ensure your dataset is correctly structured
```

## Results
After training, the script generates plots showing the accuracy and loss trends for both training and validation sets. The trained model (`wcecolonfine_tuned.h5`) can be used for inference on new images.

## Future Enhancements
- Fine-tune ResNet50 layers for better performance.
- Implement additional optimizers like **SGD with momentum**.
- Expand the dataset for improved accuracy.
- Deploy the model using **Flask** or **FastAPI** for real-time predictions.

## Acknowledgments
- **TensorFlow/Keras** for the deep learning framework.
- **ResNet50** for transfer learning.
- **Keras Tuner** for hyperparameter optimization.

---
This project is a simple yet powerful implementation of transfer learning and hyperparameter tuning for image classification. Modify and experiment with different parameters to achieve better results!

