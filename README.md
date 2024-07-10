# CIFAR-10 Image Classification

This project explores various machine learning and deep learning techniques to classify images from the CIFAR-10 dataset.

## About CIFAR-10

CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

Kaggle Dataset link:
[Kaggle](https://www.kaggle.com/c/cifar-10/)

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

![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/4ee2437c-2868-4b5b-ad53-69d0b2d5a02c)


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
   
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/2dec60d3-8218-4ef9-8c14-9cd77375a648)


2. Neural Networks:
   - Feedforward NN: 25.6% accuracy
   
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/a61a3547-286d-45fc-914b-e5f28299ef60)
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/d847c77a-33b2-40d6-a56b-49a1bbd3b7df)
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/c5f1cf11-75ae-40e2-bbc4-b16747a7494e)

   - CNN (best architecture): 93% accuracy
   
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/09f1ecf6-5064-4450-93aa-175730cb47a1)
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/d284747b-e27e-42b7-9ba9-392649faced6)
   ![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/b169cbd5-befe-47a2-9f4f-b8c16b8b7183)


   - Wide ResNet: 90%< accuracy (estimated)

## Visualizations

The project includes:
- Accuracy and loss curves for neural networks
- Confusion matrices for all models
- ROC curves and AUC scores
- Precision-Recall curves

## Make Prediction
-In the below code we use the 3rd CNN model to predict the label of the unknown test data set

![image](https://github.com/Faraz-Ardeh-2004/CIFAR10-ML/assets/59162288/596b98f3-7e71-4731-bd22-7983e7fc4cfe)

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
