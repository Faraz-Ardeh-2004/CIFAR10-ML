# CIFAR-10 Image Classification

This project explores various machine learning and deep learning techniques to classify images from the CIFAR-10 dataset.

## About CIFAR-10

CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Project Structure

1. Data Preparation
   - Loading the CIFAR-10 dataset
   - Normalizing pixel values
   - Data augmentation
   - Dimensionality reduction using PCA

2. Machine Learning Models
   - K-Nearest Neighbors (KNN)
   - Random Forest
   - Support Vector Machine (SVM)

3. Neural Networks
   - Feedforward Neural Network
   - Convolutional Neural Networks (3 different architectures)
   - Wide ResNet (Not finished due to computational limitations)

## Implementation Details

### Machine Learning Models
- Applied PCA to reduce dimensionality
- Used scikit-learn for implementation
- Evaluated using accuracy, precision, recall, F1-score, and MSE

### Neural Networks
- Implemented using TensorFlow and Keras
- Feedforward NN: 3 dense layers with dropout
- CNNs: Various combinations of Conv2D, MaxPooling2D, and Dense layers
- Wide ResNet: Based on ResNet50 architecture

## Results

1. Machine Learning Models:
   - KNN: 26% accuracy
   - Random Forest: 37% accuracy
   - SVM: 35% accuracy

2. Neural Networks:
   - Feedforward NN: 25.6% accuracy
   - CNN (best architecture): 93% accuracy
   - Wide ResNet: 90%< accuracy (estimated)

## Visualizations

The project includes:
- Accuracy and loss curves for neural networks
- Confusion matrices for all models
- ROC curves and AUC scores
- Precision-Recall curves

## How to Use

1. Install required libraries:
   tensorflow, keras, scikit-learn, numpy, pandas, matplotlib, seaborn

2. Run the Jupyter notebooks or Python scripts in this order:
   - data_preparation.py
   - machine_learning_models.py
   - neural_networks.py
   - wide_resnet.py

3. View results in the generated plots and printed metrics
4. you can change the hyperparameters too (optional)

## Conclusions
 
1. It is recommended to adjust the learning rate from 0.001 to 0.0001.
2. Increasing the number of epochs can achieve better results
3. The ratio of training data to testing and the amount of data that is checked by different algorithms and models can be effective, for example, increasing the number of input data in the training phase of the model can lead to better performance in the testing phase.
4. Mean Squared Error (MSE) is not recommended for the loss function in image classification projects, so i use loss='categorical_crossentropy'

## Future Work

- 
- Implement more advanced architectures (e.g., EfficientNet)
- Explore transfer learning
- Perform extensive hyperparameter tuning
- Investigate model interpretability techniques

## Author

Faraz Ardeh 
fa.ardeh@gmail.com
https://www.linkedin.com/in/faraz-ardeh-896917219/

This project was created as part of the 'Introductory Artificial Intelligence' course of the winter semester of SBU.
