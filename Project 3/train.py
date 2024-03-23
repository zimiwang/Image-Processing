import os
import argparse
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from utils import NoiseDataset
from torch.utils.data import DataLoader


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    trains your model for an epoch
    returns an array of loss values over the training epoch
    """
    # raise NotImplementedError("TODO: implement train_loop")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    loss_array = []
    epoch_loss = 0

    for data, label in dataloader:
        optimizer.zero_grad()

        data = data.to(DEVICE, dtype=torch.float32)
        label = label.to(DEVICE, dtype=torch.float32)

        output = model(data)
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / len(dataloader)

        print(f'loss = {loss:.8f}')
        loss_array.append(loss.item())

        # loss_array.append(epoch_loss)

    return loss_array


def test_loop(dataloader, model, loss_fn):
    """
    tests your model on the test set
    returns average MSE
    """
    # raise NotImplementedError("TODO: implement test_loop")

    test_loss = 0
    epoch_test_loss = 0

    loss_array = []

    model.eval()

    for data, label in dataloader:
        with torch.no_grad():
            data = data.to(DEVICE, dtype=torch.float32)
            label = label.to(DEVICE, dtype=torch.float32)

            output = model(data)
            loss = loss_fn(output, label)

            # test_loss += loss.item()
            loss = loss.item()
            epoch_test_loss += loss / len(dataloader)

            # loss_array.append(epoch_test_loss)
            # accuracy_array.append(epoch_test_accuracy)

            test_loss += epoch_test_loss

    print("Average MSE", test_loss)

    return test_loss


if __name__ == "__main__":
    # parse --demo flag, if not there FLAGS.demo == False
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", dest="demo", action="store_true")
    parser.set_defaults(demo=False)
    FLAGS, unparsed = parser.parse_known_args()

    # make output directory
    if not os.path.exists("./out/"):
        os.makedirs("./out/")
    # make models directory
    if not os.path.exists("./models/"):
        os.makedirs("./models/")

    # tweak these constants as you see fit, or get them through 'argparse'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    DATASET_LOC = "./nn_data/pokemon/"

    train_dataset = NoiseDataset(
        csv_file=DATASET_LOC + "training.csv",
        root_dir_noisy=DATASET_LOC + "training",
    )
    test_dataset = NoiseDataset(
        csv_file=DATASET_LOC + "testing.csv",
        root_dir_noisy=DATASET_LOC + "testing",
    )

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # TODO: define your models
    # model 1 - with single Conv2d filter
    # model = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)

    # model 2 - with five Conv2d filters
    # model = torch.nn.Sequential(
    #     torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),  # 1
    #     torch.nn.BatchNorm2d(16),
    #
    #     torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),  # 2
    #     torch.nn.BatchNorm2d(32),
    #
    #     torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),  # 3
    #     torch.nn.BatchNorm2d(64),
    #
    #     torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),  # 4
    #     torch.nn.BatchNorm2d(128),
    #
    #     torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=1),  # 5
    # )

    # model 3 - with five Conv2d filters with nonlinear layer
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),  # 1
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),  # 2
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),  # 3
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),  # 4
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=1),  # 5
    )

    # TODO: define your optimizer using learning rate and other hyperparameters
    # weight_decay normally takes 0.005
    # tweak these constants as you see fit, or get them through 'argparse'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10
    LEARNING_RATE = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    loss_fn = nn.MSELoss()  # use MSE loss for this project

    loss_total_array = []
    acc_total_array = []
    loss_epoch = []

    # FLAGS.demo = False
    if not FLAGS.demo:

        for t in range(EPOCHS):
            print(f"Epoch {t + 1}\n-------------------------------")
            loss_array = train_loop(train_dataloader, model, loss_fn, optimizer)
            loss_average = test_loop(test_dataloader, model, loss_fn)

            # Save the loss and acc into list
            loss_total_array.append(loss_array)
            loss_epoch.append(loss_average)

        print("Done!")

    # TODO: save model

    # PATH = "./models/single_linear_model.pt"
    # PATH = "./models/multiple_linear_model.pt"
    PATH = "./models/nonlinear_model_twenty_epochs.pt"
    torch.save(model, PATH)
    print("Saved model!!")

    # TODO: plot line charts of training and testing metrics (loss, accuracy)

    # Plot training loss vs iteration
    # collect the data from train and test loop
    loss_total = []
    for l in loss_total_array:
        for loss in l:
            loss_total.append(loss)

    learning_count = len(loss_total) + 1
    plt.plot(range(1, learning_count), loss_total, marker="+", label="loss")
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()

    # Plot testing error vs epoch
    plt.plot(range(1, EPOCHS + 1), loss_epoch, marker="+", label="loss")
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("epochs")
    plt.ylabel("testing error")
    plt.show()

    if FLAGS.demo:
        # TODO: set up a demo with a small subset of images
        # import the model

        # PATH = "./models/single_linear_model.pt"
        # PATH = "./models/multiple_linear_model.pt"
        # PATH = "./models/nonlinear_model_twenty_epochs.pt"

        model = torch.load(PATH)
        model.eval()
        # the number of denoised images we want
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # the number of output we want
        count = 0
        denoised_img = []
        origin_img = []

        for data, label in test_dataloader:

          data = data.to("cpu", dtype=torch.float32)
          label = label.to("cpu", dtype=torch.float32)

          prediction = model(data)

          # Convert tentor to numpy.narray for denoised img
          denoised_image = prediction.detach().numpy()
          denoised_image = np.reshape(denoised_image, (475,475))

          # Convert tentor to numpy.narray for original img

          noised_image = data.detach().numpy()
          noised_image = np.reshape(noised_image, (475,475))

          count += 1

          denoised_img.append(denoised_image)
          origin_img.append(noised_image)

          if count == 10:
            break

        # Save the images as files
        denoised_savepath = "./images/denoised/"
        noised_savepath = "./images/noised/"
        for i in range(len(denoised_img)):

          # Save the image as file for denoised img
          cv2.imwrite(os.path.join(denoised_savepath, str(i)+'.png'), denoised_img[i])

          # Save the image as file for original img
          cv2.imwrite(os.path.join(noised_savepath, str(i)+'.png'), origin_img[i])

        # TODO: plot some of the testing images by passing them through the trained model
        cv2.imshow('Noised Image', origin_img[0])
        cv2.waitKey(0)

        cv2.imshow('Denoised Image', denoised_img[0])
        cv2.waitKey(0)
