import Network

from tqdm import tqdm, trange


def main(epochs):
    model = Network.EDSR()

    # Main Loop
    for e in range(1, epochs + 1):
        print(e)




if __name__ == "__main__":
    # Parameters
    epochs = 10
    main(epochs=epochs)
