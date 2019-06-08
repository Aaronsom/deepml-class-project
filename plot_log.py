import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot(file, accuracy="sparse_categorical_accuracy"):
    log = pd.read_csv(file)
    train_loss = log["loss"]
    val_loss = log["val_loss"]
    train_acc = log[accuracy]
    val_acc = log["val_"+accuracy]

    epochs = np.arange(1, len(train_loss)+1)

    fig, ax = plt.subplots(2)
    ax[0].plot(epochs, train_loss, label="Training Loss")
    ax[0].plot(epochs, val_loss, label="Validation Loss")
    ax[0].legend()
    ax[1].plot(epochs, train_acc, label="Training Accuracy")
    ax[1].plot(epochs, val_acc, label="Validation Accuracy")
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    plot("out/log.csv")
