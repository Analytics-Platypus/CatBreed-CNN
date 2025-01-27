CNN-Based Image Classification for Multi-Class Cat Images
This project implements a Convolutional Neural Network (CNN) to classify images into one of eight categories. The script is highly configurable and optimized for generalization through techniques such as data augmentation, weight decay, dropout, and custom weight initialization.

Execution Overview
1. Data Preparation
The script uses the torchvision.datasets.ImageFolder class to load image data. The dataset is split into training and testing subsets based on the train_val_split parameter.

Training Transformations:
Applied to improve the model's generalization and handle variability in input images:
Random vertical and horizontal flips.
Random rotations up to 60 degrees.
Color jittering (adjustments to brightness, contrast, saturation, and hue) to emphasize texture and pattern recognition over color.
Testing Transformations:
No augmentations are applied to maintain consistency during evaluation.
2. Model Architecture
The CNN, defined in the Network class, consists of:

Convolutional Layers:
Five convolutional layers extract hierarchical features, with increasing depth (16 to 256 channels).
Pooling:
A max-pooling layer after the first two convolutional layers reduces feature map dimensions for computational efficiency.
Fully Connected Layers:
Two fully connected layers with dropout regularization (50%) to prevent overfitting.
Output Layer:
An 8-class output layer with a softmax activation function.
Activation Function:
ReLU is used for non-linearity in the hidden layers.
3. Custom Weight Initialization
The model uses Xavier (Glorot) uniform initialization for weights, particularly suited for layers with ReLU activations. Biases are initialized to zero.

4. Optimizer and Loss Function
Optimizer:
Adam optimizer with:
Learning rate (lr): 0.001.
Weight decay: 0.0001 (to reduce large weights and improve generalization).
Loss Function:
Cross-entropy loss, appropriate for multi-class classification problems.
5. Learning Rate Scheduler
The script optionally uses a step learning rate scheduler (StepLR) to decay the learning rate by a factor (gamma) of 0.9 every 5 epochs.

6. Training Loop
Epochs and Batches:
The training loop iterates through the dataset for a specified number of epochs (epochs) and processes batches of images (batch_size).
Training Process:
Forward pass through the network.
Loss calculation using the specified loss function.
Backpropagation to compute gradients.
Weight updates using the optimizer.
Application of the learning rate scheduler (if defined).
Metrics:
Training accuracy and loss are calculated for each epoch. Testing accuracy and a confusion matrix are printed every 10 epochs.
7. Model Evaluation
The script includes a test_network function to:

Evaluate model performance on the test dataset.
Calculate and print accuracy.
Display a confusion matrix for detailed error analysis.
8. Model Saving
Model weights are periodically saved to checkModel.pth every 10 epochs.
The final model is saved to savedModel.pth after training completes.
Key Features
Data Augmentation:
Robust training transformations for better generalization across variations in orientation, lighting, and image quality.
Custom Weight Initialization:
Improved training efficiency using Xavier initialization.
Dropout Regularization:
Added to the fully connected layers for reducing overfitting.
Learning Rate Scheduler:
Smooth decay of learning rate for fine-tuning during later epochs.
Dynamic Device Selection:
Automatic selection of CPU or GPU for computation.

Possible improvements
Skip connections to increase depth
Expanded dataset (limits on depth and data set based on computing capability)
