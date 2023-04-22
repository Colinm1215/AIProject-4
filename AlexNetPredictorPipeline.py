#Team 2 – Jarrett Arredondo, Krystal Grant, Colin Mettler, Chloé Plasse
#April 17, 2023
#CS534-S23-S01 Group Project Assignment #4 - B. Project Development

import os
import shutil
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.models.alexnet import AlexNet
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

#Runtime ~15 min
def main():

    # Setup the transform to ensure images meet the requirements for input into AlexNet.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create test and training set directories.
    dataset_path = os.path.join(os.getcwd(),"complete_mednode_dataset")
    os.makedirs(os.path.join(os.getcwd(),"train"))
    os.makedirs(os.path.join(os.getcwd(), "test"))

    # Set the paths to the training and test folders.
    train_path = os.path.join(os.getcwd(),"train")
    test_path = os.path.join(os.getcwd(),"test")

    # Create the melanoma/naevus subfolders within the test and training set directories.
    os.makedirs(os.path.join(train_path, "melanoma"))
    os.makedirs(os.path.join(train_path, "naevus"))
    os.makedirs(os.path.join(test_path, "melanoma"))
    os.makedirs(os.path.join(test_path, "naevus"))

    # Get the list of melanoma and naevus images.
    melanoma_images = os.listdir(os.path.join(dataset_path, "melanoma"))
    naevus_images = os.listdir(os.path.join(dataset_path, "naevus"))

    # Shuffle the images randomly.
    random.seed(1)
    random.shuffle(melanoma_images)
    random.shuffle(naevus_images)

    # Assign 50 melanoma images and 50 naevus images to the training set.
    for i in range(50):
        # Move melanoma image to the training set.
        src_path = os.path.join(dataset_path, "melanoma", melanoma_images[i])
        dst_path = os.path.join(train_path, "melanoma", melanoma_images[i])
        shutil.copy(src_path, dst_path)

        # Move naevus image to the training set.
        src_path = os.path.join(dataset_path, "naevus", naevus_images[i])
        dst_path = os.path.join(train_path, "naevus", naevus_images[i])
        shutil.copy(src_path, dst_path)

    # Assign the remaining 20 melanoma images and 20 of the remaining 50 naevus images to the test set.
    for i in range(50, 70):
        # Move melanoma image to the test set.
        src_path = os.path.join(dataset_path, "melanoma", melanoma_images[i])
        dst_path = os.path.join(test_path, "melanoma", melanoma_images[i])
        shutil.copy(src_path, dst_path)

    for i in range(50, 70):
        # Move naevus image to the test set.
        src_path = os.path.join(dataset_path, "naevus", naevus_images[i])
        dst_path = os.path.join(test_path, "naevus", naevus_images[i])
        shutil.copy(src_path, dst_path)

    # Load the training and test datasets.
    train_data = datasets.ImageFolder(root=train_path, transform=transform)
    test_data = datasets.ImageFolder(root=test_path, transform=transform)

    # Create dataloaders for the training and test datasets.
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

    # Convert the training and test dataloaders such that they can be fit using GridSearchCV.
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for images, labels in trainloader:
        X_train.append(images.numpy())
        y_train.append(labels.numpy())
    for images, labels in testloader:
        X_test.append(images.numpy())
        y_test.append(labels.numpy())

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Run model on GPU if available for speed.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set parameters for AlexNet training. Initialize AlexNet changing the number of classes to 2.
    # Not shown - hyperparameters chosen using GridSearchCV:
        # max_epochs = 100 (200 would be better, but factoring in run time)
        # lr = 0.001
        # optimizer = optim.SGD
        # optimizer__momentum = 0.9
        # batch_size = 10
    net = NeuralNetClassifier(
        AlexNet(num_classes=2),
        max_epochs=100,
        lr=0.001,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        criterion=nn.CrossEntropyLoss,
        device=device,
        batch_size=10
    )

    # Range of dropout rates for GridSearchCV to try (cut down total number due to run time).
    param_grid = {'module__dropout': [0.2, 0.5, 0.6, 0.7, 0.9]}

    # Set up GridSearchCV for 5-fold cross validation scoring on accuracy. Fit to the training set.
    grid = GridSearchCV(net, param_grid, scoring='accuracy', verbose=2, cv=5)
    grid.fit(X_train, y_train)

    # Print the best dropout rate.
    print('Best dropout rate: ' + str(grid.best_params_))

    # Use the best_estimator to determine accuracy on training and test sets.
    best_estimator = grid.best_estimator_
    y_pred_train = best_estimator.predict(X_train)
    y_pred = best_estimator.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test,y_pred)

    # Print accuracy scores.
    print("Training accuracy: " + str(accuracy_train*100) + "%")
    print("Test accuracy: " + str(accuracy_test*100) + "%")

if __name__ == "__main__":
    main()