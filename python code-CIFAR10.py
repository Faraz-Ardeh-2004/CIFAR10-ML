# %% [markdown]
# ### 1. Machine Learning Models

# %% [markdown]
# #### Importing essential libraries

# %%
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# %% [markdown]
# #### Spliting CIFAR10 dataset into training and test dataset

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
accuracy_list = []

# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %% [markdown]
# #### Plotting first 25 images of CIFAR10

# %%
classes = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i][0]])
plt.show()

def create_dataframe(labels, metrics):
    data = {'Metric': labels, 'Values': metrics}
    df = pd.DataFrame(data)
    return df

# Metric labels
m_labels = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'MSE']

# %% [markdown]
# #### Normalization of dataset

# %%
# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# %% [markdown]
# #### Data Augmentation and Dimensionality Reduction using PCA

# %%
# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True                         
)

# Fit the generator to the training data
datagen.fit(x_train)

# Generate augmented data
augment_size = 2000  # Adjust based on computational power
x_train_augmented = np.zeros((augment_size, 32, 32, 3))
y_train_augmented = np.zeros((augment_size, 1))

i = 0
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=augment_size, shuffle=False):
    x_train_augmented[i:i + augment_size] = x_batch
    y_train_augmented[i:i + augment_size] = y_batch
    i += augment_size
    if i >= augment_size:
        break

# Flatten the images
x_train_augmented = x_train_augmented.reshape(x_train_augmented.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reducing to 50 principal components
x_train_pca = pca.fit_transform(x_train_augmented)
x_test_pca = pca.transform(x_test)

# %% [markdown]
# ####  K-Nearest Neighbors

# %%
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

# Train the classifier
knn_classifier.fit(x_train_pca, y_train_augmented.flatten())

# %% [markdown]
# #### Calculating metrics for the evaluation of model

# %%
# predict on the test set
y_pred = knn_classifier.predict(x_test_pca)

# Calculate accuracy, precision, recall and f1 score
metrics = []

knn_accuracy = accuracy_score(y_test, y_pred) * 100
metrics.append(knn_accuracy)
precision = precision_score(y_test, y_pred, average='macro') * 100
metrics.append(precision)
recall = recall_score(y_test, y_pred, average='macro') * 100
metrics.append(recall)
f1 = f1_score(y_test, y_pred, average='macro') * 100
metrics.append(f1)
MSE = mean_squared_error(y_test, y_pred)
metrics.append(MSE)
df = create_dataframe(m_labels, metrics)
print(df)
accuracy_list.append(knn_accuracy)

# %% [markdown]
# #### Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Calssifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(x_train_pca, y_train_augmented.flatten())

# %%
# predict on the test set
y_pred = rf_classifier.predict(x_test_pca)

# Calculate accuracy, precision, recall and f1 score
metrics = []

rf_accuracy = accuracy_score(y_test, y_pred) * 100
metrics.append(rf_accuracy)
precision = precision_score(y_test, y_pred, average='macro') * 100
metrics.append(precision)
recall = recall_score(y_test, y_pred, average='macro') * 100
metrics.append(recall)
f1 = f1_score(y_test, y_pred, average='macro') * 100
metrics.append(f1)
MSE = mean_squared_error(y_test, y_pred)
metrics.append(MSE)
df = create_dataframe(m_labels, metrics)
print(df)
accuracy_list.append(rf_accuracy)

# %% [markdown]
# #### Support Vector Machine

# %%
# Train a linear SVM
svm_model = SVC(kernel='linear')

# Train the SVM model
svm_model.fit(x_train_pca, y_train_augmented.flatten())

# %%
# Predict on the test set
y_pred = svm_model.predict(x_test_pca)

# Calculate accuracy, precision, recall and f1 score
metrics = []

svm_accuracy = accuracy_score(y_test, y_pred) * 100
metrics.append(svm_accuracy)
precision = precision_score(y_test, y_pred, average='macro') * 100
metrics.append(precision)
recall = recall_score(y_test, y_pred, average='macro') * 100
metrics.append(recall)
f1 = f1_score(y_test, y_pred, average='macro') * 100
metrics.append(f1)
MSE = mean_squared_error(y_test, y_pred)
metrics.append(MSE)
df = create_dataframe(m_labels, metrics)
print(df)
accuracy_list.append(svm_accuracy)

# %% [markdown]
# #### Performance Visualization

# %%
accuracy_list = accuracy_list / sum(accuracy_list)
mylabels = ["KNN", "Random Forest", "SVM"]
mycolors = ["#A2D9CE", "#F9E79F", "#D6DBDF"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.pie(accuracy_list, labels = mylabels, colors=mycolors)
ax1.set_title("Pie Chart of different ML model accuracy")

ax2.bar(mylabels, accuracy_list, color=mycolors)
ax2.set_ylabel('Accuracy')
ax2.set_title('Bar Chart of different ML model accuracy')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2. Neural Networks

# %% [markdown]
# #### Import Neccesary libraries for implementation of Neural Networks

# %%
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall, MeanSquaredError
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# %% [markdown]
# #### Performance Visualization
# ##### Below cell is implemention of different metric and curve which are visualized

# %%
# Accuracy and Loss curve
def plot_accuracy_and_loss(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ROC curve and AUC
def plot_roc_curve(y_true, y_pred_proba, classes):
    plt.figure(figsize=(12, 4))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {classes[i]} (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

# Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred_proba, classes):
    plt.figure(figsize=(10,4))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        plt.plot(recall, precision, label=f"Class {classes[i]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.show()

# %% [markdown]
# #### Feedforward Neural Network

# %% [markdown]
# ##### Building the architecture of FF neural network

# %%
# Build the model
ff_model = Sequential()
# Flatten the input
ff_model.add(Flatten(input_shape=(32, 32, 3)))
ff_model.add(Dense(512, activation='relu'))
ff_model.add(Dropout(0.5))
ff_model.add(Dense(512, activation='relu'))
ff_model.add(Dropout(0.5))
ff_model.add(Dense(512, activation='relu'))
ff_model.add(Dropout(0.5))

# Output layer
ff_model.add(Dense(10, activation='softmax'))

# %% [markdown]
# ##### Compilation of the model

# %%
# Compile the model
ff_model.compile(optimizer=Adam(),
                 loss="sparse_categorical_crossentropy",
                 metrics = ["accuracy"]
                )

# %% [markdown]
# ##### Model Summary

# %%
print(ff_model.summary())

# %% [markdown]
# ##### Training the Model

# %%
# Train the model
ff_history = ff_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=2)

# %% [markdown]
# ##### Evalution of the model

# %%
# Evaluate the model
test_loss, test_acc = ff_model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100: .2f}%")

# predict on the test set
y_pred = np.argmax(ff_model.predict(x_test), axis=1)
# Calculate other metrics
precision = precision_score(y_test, y_pred, average='macro', zero_division=0.0)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")

# %% [markdown]
# ##### Plotting the accracy and loss curve of the model

# %%
plot_accuracy_and_loss(ff_history)

# %% [markdown]
# ##### Confusion Matrix of the model

# %%
plot_confusion_matrix(y_test, y_pred, "Confusion matrix of FF neural network")

# %% [markdown]
# ##### ROC curve and AUC

# %%
y_pred_proba = tf.nn.softmax(ff_model.predict(x_test)).numpy()
plot_roc_curve(y_test, y_pred_proba, classes=range(10))

# %% [markdown]
# ##### Plotting the Precision-Recall curve

# %%
plot_precision_recall_curve(y_test, y_pred_proba, classes=range(10))

# %% [markdown]
# #### Convolutional Neural Network
# ##### In this section we build, train and evaluate multiple CNN with various parameter like number and configuration of layers,
# ##### loss function, metric, optimizer and different values for hyperparameters.

# %%
# Build the model
model_1 = Sequential()
model_1.add(Conv2D(filters=32, kernel_size=(3, 3), activation="tanh", input_shape=(32, 32, 3)))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Conv2D(filters=64, kernel_size=(4, 4), activation="tanh"))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Flatten())
model_1.add(Dense(units=34, activation="tanh"))
model_1.add(Dense(units=10, activation="softmax"))

# %%
# Compile model
model_1.compile(
    optimizer=SGD(),
    loss="sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

# %%
# Train the model
history_1 = model_1.fit(x_train, y_train, batch_size=64, validation_data=(x_test, y_test), epochs=10)

# %%
# Evaluate the model
test_loss, test_acc = model_1.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100: .2f}%")

# predict on the test set
y_pred = np.argmax(model_1.predict(x_test), axis=1)

# Calculate other metrics
precision = precision_score(y_test, y_pred, average='macro',zero_division=0.0)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")

# %%
plot_accuracy_and_loss(history_1)

# %%
plot_confusion_matrix(y_test, y_pred, "Confusion matrix of convolutional neural network")

# %%
y_pred_proba = tf.nn.softmax(model_1.predict(x_test)).numpy()
plot_roc_curve(y_test, y_pred_proba, classes=range(10))

# %%
plot_precision_recall_curve(y_test, y_pred_proba, classes=range(10))

# %% [markdown]
# ### Another architecture for CNN model

# %%
# Building the model
model_2 = Sequential()
model_2.add(Conv2D(filters=64, kernel_size=(3, 3), activation="sigmoid", input_shape=(32, 32, 3)))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.2)) 
model_2.add(Conv2D(filters=128, kernel_size=(3, 3), activation="sigmoid"))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.2)) 
model_2.add(Flatten())
model_2.add(Dense(units=64, activation="sigmoid"))
model_2.add(Dense(units=10, activation="softmax"))

# %%
# Compile the model
model_2.compile(optimizer=RMSprop(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# %%
# Train the model
history_2 = model_2.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)

# %%
# Evaluate the model
test_loss, test_acc = model_2.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100: .2f}%")

# predict on the test set
y_pred = np.argmax(model_2.predict(x_test), axis=1)

# Calculate other metrics
precision = precision_score(y_test, y_pred, average='macro',zero_division=0.0)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")

# %%
plot_accuracy_and_loss(history_2)

# %%
plot_confusion_matrix(y_test, y_pred, "Confusion matrix of Convolutional neural network 2")

# %%
y_pred_proba = tf.nn.softmax(model_2.predict(x_test)).numpy()
plot_roc_curve(y_test, y_pred_proba, classes=range(10))

# %%
plot_precision_recall_curve(y_test, y_pred_proba, classes=range(10))

# %%
# Building the model
model_3 = Sequential()
model_3.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)))
model_3.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model_3.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Flatten())
model_3.add(Dense(units=512, activation="relu"))
model_3.add(Dropout(0.5))          
model_3.add(Dense(units=10, activation="softmax"))

# %%
# Compile the model
model_3.compile(optimizer=Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# %%
# Train the model
history_3 = model_3.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=2)

# %%
# Evaluate the model
test_loss, test_acc = model_3.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100: .2f}%")

# predict on the test set
y_pred = np.argmax(model_3.predict(x_test), axis=1)

# Calculate other metrics
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")

# %%
plot_accuracy_and_loss(history_3)

# %%
plot_confusion_matrix(y_test, y_pred, "Confusion matrix of Convolutional neural network 3")

# %%
y_pred_proba = tf.nn.softmax(model_3.predict(x_test)).numpy()
plot_roc_curve(y_test, y_pred_proba, classes=range(10))

# %%
plot_precision_recall_curve(y_test, y_pred_proba, classes=range(10))

# %% [markdown]
# #### Make Prediction
# ##### In the below code we use model 3 to predict the label of unknown test data set

# %%
def show_image(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.grid(False)
    plt.imshow(x[index])
    plt.xlabel(classes[int(y[index])])

# %%
show_image(x_test, y_test, 3)

# %%
y_predictions = model_3.predict(x_test)
y_predictions = [np.argmax(arr) for arr in y_predictions]
print(classes[y_predictions[3]])

# %% [markdown]
# ### Wide ResNet

# %%
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# %%
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %%
# Build a Wide ResNet Model

input_shape = (32, 32, 3)
num_classes = 10
base_rn_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
x = Flatten()(base_rn_model.output)
x = Dense(num_classes, activation="softmax")(x)
rn_model = tf.keras.models.Model(inputs=base_rn_model.input, outputs=x)

# %%
# Compile the model
rn_model.compile(Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy']
                )

# %%
# Train the model
history = rn_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# %%



