#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import KernelPCA


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images # n_samples, 8, 8
labels = digits.target # n_samples

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list, images, labels):
  # insert code that when given a list of integers, will find the labels and images
  # and put them all in numpy arrary (at the same time, as training and testing data)
    if not isinstance(number_list, list):
        raise TypeError("number_list must be a list")
    
    images_list = []
    labels_list = []
    for num in number_list:
        if num not in range(10):  # only digit
            raise ValueError(f"Invalid digit {num}. Must be between 0-9")
        indices = np.where(labels == num)[0]
        images_list.extend(images[indices])
        labels_list.extend(labels[indices])
    return np.array(images_list), np.array(labels_list)

def print_numbers(images, labels):
  # insert code that when given images and labels (of numpy arrays)
  # the code will plot the images and their labels in the title.
    max_cols = 20
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must be the same.")
    
    n = len(images)
    rows = (n+max_cols-1) // max_cols  # Display up to 10 images per row
    cols = min(n, max_cols)  # Up to 10 columns per row
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))

    for i, (image, label) in enumerate(zip(images, labels)):
        ax = axes[i // max_cols, i % max_cols]  # Access the subplot for the current image
        ax.imshow(image, cmap='gray') # Display the image in grayscale
        ax.axis('off')
    
    # Turn off unused axes
    for j in range(i + 1, rows * cols):
        ax = axes[j // max_cols, j % max_cols]
        ax.axis('off')
    try:
        plt.tight_layout(pad=0.1)
    except ValueError as e:
        print(f"Warning: Unable to apply tight layout: {e}")
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    plt.show()

def evaluate_model(model, X_test, y_test, model_name=""):
    y_pred = model.predict(X_test)
    accuracy = OverallAccuracy(y_pred, y_test)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    return accuracy

def denoise_data(X_train_poison, n_components=64):
    X_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
    kpca = KernelPCA(
        n_components=n_components, 
        kernel='rbf',
        gamma=0.1,
        fit_inverse_transform=True  # 允许反转变换
    )
    X_denoised = kpca.fit_transform(X_reshaped)
    return X_denoised

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels )


model_1 = GaussianNB()

# however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
# Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test_reshaped)


def OverallAccuracy(results, actual_values):
  # Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  correct = np.sum(results == actual_values)
  total = len(actual_values)
  Accuracy =  correct / total
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


# Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, X_test, y_test)



# Part 6
# Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall accuracy of the KNN model is " + str(Model2_Overall_Accuracy))

# Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0,max_iter=500)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall accuracy of the MLP model is " + str(Model3_Overall_Accuracy))


# Part 8
# Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


# Part 9-11
# Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
# Poisoned data
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)

# GaussianNB with poisoned data
model_1.fit(X_train_poison_reshaped, y_train)
model1_poisoned_results = model_1.predict(X_test_reshaped)
Model1_Poisoned_Accuracy = OverallAccuracy(model1_poisoned_results, y_test)
print("Gaussian model accuracy on poisoned data: " + str(Model1_Poisoned_Accuracy))

# KNeighborsClassifier with poisoned data
model_2.fit(X_train_poison_reshaped, y_train)
model2_poisoned_results = model_2.predict(X_test_reshaped)
Model2_Poisoned_Accuracy = OverallAccuracy(model2_poisoned_results, y_test)
print("KNN model accuracy on poisoned data: " + str(Model2_Poisoned_Accuracy))

# MLPClassifier with poisoned data
model_3.fit(X_train_poison_reshaped, y_train)
model3_poisoned_results = model_3.predict(X_test_reshaped)
Model3_Poisoned_Accuracy = OverallAccuracy(model3_poisoned_results, y_test)
print("MLP model accuracy on poisoned data: " + str(Model3_Poisoned_Accuracy))


# Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

#X_train_denoised = # fill in the code here

# Denoise poisoned data
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
kpca = KernelPCA(n_components=64, kernel='rbf', gamma=0.1)
X_train_denoised = kpca.fit_transform(X_train_poison_reshaped)

# Train GaussianNB with denoised data
model_1.fit(X_train_denoised, y_train)
model1_denoised_results = model_1.predict(X_test_reshaped)
Model1_Denoised_Accuracy = OverallAccuracy(model1_denoised_results, y_test)
print("Gaussian model accuracy on denoised data: " + str(Model1_Denoised_Accuracy))

# Train KNN with denoised data
model_2.fit(X_train_denoised, y_train)
model2_denoised_results = model_2.predict(X_test_reshaped)
Model2_Denoised_Accuracy = OverallAccuracy(model2_denoised_results, y_test)
print("KNN model accuracy on denoised data: " + str(Model2_Denoised_Accuracy))

# Train MLPClassifier with denoised data
model_3.fit(X_train_denoised, y_train)
model3_denoised_results = model_3.predict(X_test_reshaped)
Model3_Denoised_Accuracy = OverallAccuracy(model3_denoised_results, y_test)
print("MLP model accuracy on denoised data: " + str(Model3_Denoised_Accuracy))


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
performance_table = {
    "Model": ["GaussianNB", "KNeighborsClassifier", "MLPClassifier"],
    "Clean Data Accuracy": [
        Model1_Overall_Accuracy,
        Model2_Overall_Accuracy,
        Model3_Overall_Accuracy,
    ],
    "Poisoned Data Accuracy": [
        Model1_Poisoned_Accuracy,
        Model2_Poisoned_Accuracy,
        Model3_Poisoned_Accuracy,
    ],
    "Denoised Data Accuracy": [
        Model1_Denoised_Accuracy,
        Model2_Denoised_Accuracy,
        Model3_Denoised_Accuracy,
    ],
}

import pandas as pd
performance_df = pd.DataFrame(performance_table)
print(performance_df)

performance_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()