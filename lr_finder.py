import Network
import Utils

from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np

import multiprocessing as mp
from tqdm import tqdm
import os

### Note ###
"""
To use learning_rate_finder, you must use an unbounded loss function. Loss functions that are bounded by [0, 1] do not create proper graphs
"""


def lr_finder(dataset_path, start_lr, end_lr, iterations):
    # Parameters
    lr_factor = np.exp(np.log(end_lr / start_lr) / iterations)
    batch_size = 4
    plot_iter = 100
    search_iter = 1

    # Get Model
    model = Network.DENSESR()
    model.summary()

    # Get Image Paths
    dataset_paths = list(os.listdir(dataset_path))
    dataset_paths = np.array(
        [f"{dataset_path}{image_path}" for image_path in dataset_paths]
    )

    # Multiprocessing
    workers = mp.cpu_count()
    if batch_size <= mp.cpu_count():
        workers = batch_size
    workers = 1
    print(f"Workers: {workers}")
    if workers != 1:
        p = mp.Pool(workers)

    K.set_value(model.optimizer.learning_rate, start_lr)

    metrics = []
    lr = start_lr
    # Main Loop
    for i in tqdm(range(iterations)):
        temp_loss = 0

        for _ in range(search_iter):
            # Load Batch
            image_paths = dataset_paths[
                np.random.randint(0, len(dataset_paths), size=(batch_size))
            ]

            # Load Data
            if workers != 1:
                X, y = Utils.load_multiprocessing(p, image_paths)
            else:
                X, y = Utils.load(image_paths)

            # Get Loss
            temp_loss += model.train_on_batch(X, y)[0]

        loss = temp_loss / search_iter

        metrics.append([i, lr, loss])

        lr *= lr_factor

        K.set_value(model.optimizer.learning_rate, lr)

        if i % plot_iter == 0:
            plot(np.array(metrics), show=False)

    return np.array(metrics)


def plot(metrics, show):
    iterations = metrics[:, 0]
    lr = metrics[:, 1]
    loss = metrics[:, 2]

    fig, ax1 = plt.subplots()

    # x: Iterations, y: Learning Rate
    ax1.plot(iterations, lr)
    ax1.set_yscale("log")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("learning rate")

    # x: Iterations, y: Loss
    ax2 = ax1.twinx()
    ax2.plot(iterations, loss, color="green")
    ax2.set_yscale("log")
    ax2.set_ylabel("loss")

    # Save Plot
    plt.savefig("plot_resnet_50.png")

    # Display Plot
    if show:
        plt.show()

    # Close Plot
    plt.close()


def main(search):
    # Get Dataset Path
    dataset_path = "data/train/DIV2K_FLICKR2K/"

    ### Training Loop ###
    if search:
        metrics = lr_finder(
            dataset_path=dataset_path, start_lr=1e-10, end_lr=1, iterations=1000
        )
        np.save("metrics", metrics)
    else:

        metrics = np.load("metrics.npy")

    plot(metrics, show=False)


if __name__ == "__main__":
    main(search=True)
