import Utils
from generator import DataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import datetime


load_path = 'data/train/training_HR/'
save_path = 'weights/VDSR/temp/'
def main():
    # Initialize Constants
    BS = 32

    # Initialize Model
    model = Utils.model_large_SRCNN()
    print(model.summary())

    # Initialize Generator
    file_names = np.array(os.listdir(load_path))    
    training_generator = DataGenerator(file_names, load_path, batch_size=BS, dim=(64,64), scale=[4, 3, 2], n_channels=3)

    # Load Existing Model
    #model.load_weights()

    # Callbacks
    model_checkpoint_callback = ModelCheckpoint(filepath=save_path + 'weights.{epoch:02d}-{loss:.10f}.hdf5',
                                                monitor='loss',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode = 'min',
                                                save_freq='epoch')
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.5,
                                  verbose=1,
                                  patience=4,
                                  min_delta=0)
    early_stop = EarlyStopping(monitor='loss',
                               min_delta=0,
                               patience=16,
                               verbose=1,
                               restore_best_weights=True)
    csv_log = CSVLogger('weights/VDSR/temp/VDSR_9x64-3_3-3.csv',
                        append=True)

    # Model training
    CNN = model.fit(training_generator,
                    epochs=300,
                    initial_epoch=0,
                    use_multiprocessing=True,
                    workers=8,
                    steps_per_epoch=1000,
                    callbacks=[model_checkpoint_callback, reduce_lr, early_stop, csv_log])

    # Initialize History Plot Points
    training_loss = CNN.history['loss']
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    fig = plt.figure(figsize=(10,5))
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(save_path + 'training_plot.jpg', bbox_inches='tight', dpi=150)
    #plt.show()
    
    model.save_weights(filepath = save_path + 'final_weights.h5')
if __name__ == '__main__':
    main()
