#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from albumentations.pytorch import ToTensorV2

from app.models.resnet import ResNet
from app.utils.transforms import transforms
from app.datasets.cifar10 import load_cifar10
from app.explainability.gradcam import GradCAM
from app.utils.misc import set_seed, tensor_to_image
from app.utils.train_test_loops import train_loop, test_loop
from app.utils.result_stats import loss_acc_curves, score_report
from app.utils.result_analysis import (
    get_misclassified_info,
    visualize_misclassified_images,
)
from app.utils.dataset import (
    calc_data_stats,
    visualise_transforms,
    class_to_idx,
    idx_to_class,
)

import matplotlib.pyplot as plt

set_seed(69)


################################
## CONFIG
################################
DATASET_NAME = "cifar10"
DATASET_PATH = f"./../data/{DATASET_NAME}/"
NUM_CLASSES = 10
NUM_INPUT_CHANNELS = 3


BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.01


# SET GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nWe're using =>", device)


################################
## LOAD DATASET
################################

# datasets
og_dataset = load_cifar10(
    dataset_path=DATASET_PATH, is_train=False, image_transforms=ToTensorV2(),
)

# calculate dataset mean and std
dataset_mean, dataset_std = calc_data_stats(og_dataset)

image_transforms = transforms(dataset_mean, dataset_std)

train_dataset = load_cifar10(
    dataset_path=DATASET_PATH,
    is_train=True,
    image_transforms=image_transforms["train"],
)

test_dataset = load_cifar10(
    dataset_path=DATASET_PATH,
    is_train=False,
    image_transforms=image_transforms["test"],
)



class2idx = class_to_idx(og_dataset.classes)
idx2class = idx_to_class(class2idx)




visualise_transforms(
    original_data=og_dataset, transformed_data=train_dataset, num_samples=10
)




################################
## CREATE DATALOADERS
################################

# dataloader
train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE)

val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE)

test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE)

## Data Sanity Check
print(f"\nTrain loader = {next(iter(train_loader))[0].shape}")
print(f"Val loader = {next(iter(val_loader))[0].shape}")
print(f"Test loader = {next(iter(test_loader))[0].shape}")
print(f"\nTrain loader length = {len(train_loader)}")
print(f"Val loader length = {len(val_loader)}")
print(f"Test loader length = {len(test_loader)}")





################################
## LOAD MODEL
################################

model = ResNet(num_input_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES)

x_train_example, y_train_example = next(iter(train_loader))
y_pred_example = model(x_train_example)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)





################################
## Train Loop
################################
trained_model, loss_stats, acc_stats = train_loop(
    model=model,
    epochs=EPOCHS,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
)


loss_acc_curves(loss_stats=loss_stats, acc_stats=acc_stats)




################################
## Test Loop
################################
y_pred_list, y_true_list = test_loop(
    model=trained_model, test_loader=test_loader, device=device,
)





################################
## Result Stats
################################
print(score_report(y_true_list, y_pred_list, idx2class))





################################
## Result Analysis
################################


def get_misclassified_info(y_pred_list, y_true_list):
    return [
        {"idx": i, "pred": pred, "true": actual}
        for i, (pred, actual) in enumerate(zip(y_pred_list, y_true_list))
        if pred != actual
    ]


misclassified_info = get_misclassified_info(
    y_pred_list=y_pred_list, y_true_list=y_true_list
)





visualize_misclassified_images(
    misclassified_info=misclassified_info,
    dataset=og_dataset,
    idx_to_class=idx2class,
    num_samples=11,
)





x_test, y_test = test_dataset[40]





tensor_to_image(x_test)




cam = GradCAM(
    model=model, target_layer=trained_model.layer_2.base_layer[-1].base_block[-1][0]
)
output = cam(x_test)





plt.imshow(output)




