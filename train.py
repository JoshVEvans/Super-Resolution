import utils
import output
import network

from keras import backend as K
from tensorflow.keras.utils import plot_model

import os
import importlib
import numpy as np
import multiprocessing as mp
from tqdm import tqdm, trange


def main():
    # Parameters
    dataset_path = "data/COCO/"
    batch_size = 16
    epochs = 500
    steps_per_epoch = 1000
    workers = 4

    ### Training Loop ###
    training(
        dataset_path=dataset_path,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        workers=workers,
    )


def training(dataset_path, batch_size, epochs, steps_per_epoch, workers):
    # Metrics
    saved_loss = float("inf")
    lr_decay_num = 20
    best_epoch = 0
    temp = 0

    # Get Model
    model = network.SRCNN()
    model.summary()
    plot_model(model, to_file="model.png")

    # Multiprocessing
    print(f"Workers: {workers}")
    if workers != 1:
        p = mp.Pool(workers)

    # Get Image Paths
    dataset_paths = list(os.listdir(dataset_path))
    dataset_paths = np.array(
        [f"{dataset_path}{image_path}" for image_path in dataset_paths]
    )

    # Main Loop
    for e in tqdm(range(1, epochs + 1)):
        # Metrics
        avg_loss = []
        avg_accuracy = []

        # Create Progress Bar
        tr = trange(steps_per_epoch, desc="Epoch: ", leave=True)
        for i in tr:
            # Get image paths
            image_paths = dataset_paths[
                np.random.randint(0, len(dataset_paths), size=(batch_size))
            ]

            # Load Images
            if workers != 1:
                X, y = utils.load_multiprocessing(p, image_paths)
            else:
                X, y = utils.load(image_paths)

            # Train Model
            metrics = model.train_on_batch(X, y)

            # Update Metrics
            avg_loss.append(metrics[0])
            avg_accuracy.append(metrics[1])

            # Update Progress Bar
            tr.set_description(f"Epoch-{e}/{epochs}")
            tr.set_postfix(
                loss=np.array(avg_loss).mean(), accuracy=np.array(avg_accuracy).mean()
            )

        # Metrics
        avg_loss = np.array(avg_loss).mean()
        avg_accuracy = np.array(avg_accuracy).mean()

        ### Learning Rate Decay ###
        if e != 1:
            if saved_loss < avg_loss:
                print(
                    f"Best Epoch({best_epoch}) loss({saved_loss}) was better than current({avg_loss})"
                )
                temp += 1
            else:
                temp = 0

            # Update Learning Rate
            if temp >= lr_decay_num:
                best_epoch = e
                saved_loss = avg_loss
                K.set_value(
                    model.optimizer.learning_rate, model.optimizer.learning_rate / 2
                )
                temp = 0
                print(f"Learning Rate Updated to {model.optimizer.learning_rate}")
        if saved_loss > avg_loss:
            best_epoch = e
            saved_loss = avg_loss

        ### Save Weights ###
        model.save(f"weights/{e}-{avg_loss}-{avg_accuracy}.h5")
        model.save(f"weights/weights.h5")

        ### Test Model ###
        # Reload Module
        importlib.reload(output)
        output.evaluate(model, summary=False)


if __name__ == "__main__":
    main()
