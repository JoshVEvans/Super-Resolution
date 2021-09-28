import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


weight_folders = [
    ["large", "weights_large/"],
    ["small", "weights_small/"],
    ["SRCNN", "weights_SRCNN/"],
    ["VDSR", "weights_VDSR/"],
]


for weight_folder in weight_folders:
    name = weight_folder[0]
    directory = weight_folder[1]

    weight_paths = np.array(
        [f"{directory}{weight_path}" for weight_path in os.listdir(directory)]
    )

    epochs = []
    loss = []
    accuracy = []
    df = pd.DataFrame()

    for weight_path in weight_paths:
        split_weight = weight_path.split("-")

        epochs.append(float(split_weight[0].split("/")[1]))
        loss.append(float(split_weight[1]))
        accuracy.append(float(f'0.{split_weight[2].split(".")[1]}'))

    df["epochs"] = epochs
    df["loss"] = loss
    df["accuracy"] = accuracy
    df.sort_values(by=["epochs"], inplace=True)

    plt.plot(df.epochs, df.loss.rolling(25).mean(), label=name)
    # print(loss)
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('plot.png')
plt.show()