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
        images_list.append(images[indices[0]])
        labels_list.append(labels[indices[0]])
    return np.array(images_list), np.array(labels_list)

def print_numbers(images, labels):
  # insert code that when given images and labels (of numpy arrays)
  # the code will plot the images and their labels in the title.
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must be the same.")
    
    n = len(images)
    rows = 1  #isplay up to 10 images per row
    cols = min(n, 5)  # Up to 10 columns per row
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))

    for i, (image, label) in enumerate(zip(images, labels)):
        ax = axes[i % 5]  # Access the subplot for the current image
        ax.imshow(image, cmap='gray') # Display the image in grayscale
        ax.axis('off')
    
    # Turn off unused axes
    for j in range(i + 1, rows * cols):
        ax = axes[j % 5]
        ax.axis('off')
    try:
        plt.tight_layout(pad=0.1)
    except ValueError as e:
        print(f"Warning: Unable to apply tight layout: {e}")
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    plt.show()


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


# Part 8: Poisoning the Training Data
# Add random Gaussian noise to the training data to simulate poisoning
noise_scale = 0.25
poison = rng.normal(scale=noise_scale, size=X_train.shape)  # Generate noise
X_train_poison = X_train + poison  # Add noise to the training images

# Reshape poisoned data for model training
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)

# GaussianNB with poisoned data
model_1.fit(X_train_poison_reshaped, y_train)  # Train on poisoned data
model1_poisoned_results = model_1.predict(X_test_reshaped)  # Predict on clean test data
Model1_Poisoned_Accuracy = OverallAccuracy(model1_poisoned_results, y_test)  # Calculate accuracy
print("Gaussian model accuracy on poisoned data: " + str(Model1_Poisoned_Accuracy))

# KNeighborsClassifier with poisoned data
model_2.fit(X_train_poison_reshaped, y_train)  # Train on poisoned data
model2_poisoned_results = model_2.predict(X_test_reshaped)  # Predict on clean test data
Model2_Poisoned_Accuracy = OverallAccuracy(model2_poisoned_results, y_test)  # Calculate accuracy
print("KNN model accuracy on poisoned data: " + str(Model2_Poisoned_Accuracy))

# MLPClassifier with poisoned data
model_3.fit(X_train_poison_reshaped, y_train)  # Train on poisoned data
model3_poisoned_results = model_3.predict(X_test_reshaped)  # Predict on clean test data
Model3_Poisoned_Accuracy = OverallAccuracy(model3_poisoned_results, y_test)  # Calculate accuracy
print("MLP model accuracy on poisoned data: " + str(Model3_Poisoned_Accuracy))



# Part 12-13: Denoising the Poisoned Data
# Reshape poisoned data for KernelPCA
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)

# Apply KernelPCA for denoising
# Reduce dimensions and enable inverse transformation for visualization
kpca = KernelPCA(n_components=64, kernel='linear', gamma=0.001, fit_inverse_transform=True)
X_train_denoised = kpca.fit_transform(X_train_poison_reshaped)  # Denoised training data

# Transform the test data to match the reduced dimensions
X_test_denoised = kpca.transform(X_test_reshaped)

# GaussianNB with denoised data
model_1.fit(X_train_denoised, y_train)  # Train on denoised data
model1_denoised_results = model_1.predict(X_test_denoised)  # Predict on transformed test data
Model1_Denoised_Accuracy = OverallAccuracy(model1_denoised_results, y_test)  # Calculate accuracy
print("Gaussian model accuracy on denoised data: " + str(Model1_Denoised_Accuracy))

# KNN with denoised data
model_2.fit(X_train_denoised, y_train)  # Train on denoised data
model2_denoised_results = model_2.predict(X_test_denoised)  # Predict on transformed test data
Model2_Denoised_Accuracy = OverallAccuracy(model2_denoised_results, y_test)  # Calculate accuracy
print("KNN model accuracy on denoised data: " + str(Model2_Denoised_Accuracy))

# MLPClassifier with denoised data
model_3.fit(X_train_denoised, y_train)  # Train on denoised data
model3_denoised_results = model_3.predict(X_test_denoised)  # Predict on transformed test data
Model3_Denoised_Accuracy = OverallAccuracy(model3_denoised_results, y_test)  # Calculate accuracy
print("MLP model accuracy on denoised data: " + str(Model3_Denoised_Accuracy))

# Part 14-15: Performance Comparison
# Create a table summarizing the performance of the models
performance_table = {
    "Model": ["GaussianNB", "KNeighborsClassifier", "MLPClassifier"],
    "Clean Data Accuracy": [
        Model1_Overall_Accuracy,   # Accuracy of clean data
        Model2_Overall_Accuracy,
        Model3_Overall_Accuracy,
    ],
    "Poisoned Data Accuracy": [
        Model1_Poisoned_Accuracy,  # Accuracy of poisoned data
        Model2_Poisoned_Accuracy,
        Model3_Poisoned_Accuracy,
    ],
    "Denoised Data Accuracy": [
        Model1_Denoised_Accuracy,  # Accuracy of denoised data
        Model2_Denoised_Accuracy,
        Model3_Denoised_Accuracy,
    ],
}

# Convert the performance table to a DataFrame for better display
import pandas as pd
performance_df = pd.DataFrame(performance_table)
print(performance_df)

# Plot performance comparison as a bar chart
performance_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Visualize the first few reconstructed images
reconstructed_images = kpca.inverse_transform(X_train_denoised).reshape(X_train.shape)

def visualize_reconstructed(images, original, noised, indices):
    fig, axes = plt.subplots(3, len(indices), figsize=(15, 8))  # Increased height for clarity

    for i, idx in enumerate(indices):
        # Original image
        axes[0, i].imshow(original[idx], cmap='gray')
        axes[0, i].axis('off')  # Remove axis labels for cleaner visuals

        # Noisy image
        axes[1, i].imshow(noised[idx], cmap='gray')
        axes[1, i].axis('off')

        # Reconstructed image
        axes[2, i].imshow(images[idx], cmap='gray')
        axes[2, i].axis('off')

    # Add row titles
    fig.text(0.5, 0.9, "Original Images", ha='center', fontsize=14)  # Title for the first row
    fig.text(0.5, 0.6, "Poisoned Images", ha='center', fontsize=14)  # Title for the second row
    fig.text(0.5, 0.3, "Denoised Images", ha='center', fontsize=14)  # Title for the third row

    plt.tight_layout()
    plt.show()

# Choose indices to visualize
indices_to_visualize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
visualize_reconstructed(reconstructed_images, X_train, X_train_poison, indices_to_visualize)