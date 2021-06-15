import gc

import keras
import tensorflow as tf

import helper_train_mean as helper
# import helper

# import posenet_dropout as posenet
# import posenet_dropout_regu as posenet
import posenetLSTMDropout as posenet
# import posenetLSTMnoSequence as posenet
# import posenetLSTM as posenet
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import random
from generator import DataGenerator
from keras.utils.vis_utils import plot_model

# define the setting of the training. Note some other setting of the network can be found in the begenning of the posenetLSTMDropout.py file
window = 4
learningRate = 0.001
batch_size = 25
beta = posenet.beta
LSTM_size = posenet.LSTM_size
drop1 = posenet.drop1
drop2 = posenet.drop2


# def rmse(y_true, y_pred):
#    from keras import backend
#    return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

# function to check the loss of the last branch of the network
class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y, z = self.test_data
        one, two, three, four, five, six = self.model.predict(x, verbose=0)
        diff_sum = np.sum(np.absolute(np.squeeze(y) - np.squeeze(five)))
        print(diff_sum / 25)
        return diff_sum / 25


#        print('\nTesting loss: {}, acc: {}\n'.format(difference, difference))

# function to retrn the loss
def validation_error_x(y_true, y_pred):
    loss_x = keras.backend.sum(keras.backend.abs(y_true - y_pred))
    return loss_x


if __name__ == "__main__":
    current_epoch = 77
    model = posenet.create_posenet('/home/tarekfourati/pfa/window10batch16LR0.001beta600LSTM256Dropout0.250.25.77.h5', True, window)

    # for layer in model.layers:
    #     # layer.trainable = False
    #     if isinstance(layer, keras.layers.normalization.BatchNormalization):
    #         layer._per_input_updates = {}
    #         print("BATCH NORM FROZEN")
    #     print(layer.name)
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # define optimiser, compile model and get the training dataset
    adam = Adam(lr=learningRate, clipvalue=1.5)

    # compile the model with the custom loss function
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        #                                        'cls3_fc_pose_xyz_new': posenet.euc_loss3x, 'cls3_fc_pose_wpqr_new': posenet.euc_loss3q})
                                        'cls3_fc_pose_xyz_new': posenet.euc_loss3x,
                                        'cls3_fc_pose_wpqr_new': posenet.euc_loss3q}, metrics=[validation_error_x])

    directory = '/home/tarekfourati/pfa/RecurrentBIM-PoseNet/RecurrentBIMPoseNetDataset/Synthetic dataset/Gradmag-Syn-Car/'
    dataset_train = 'groundtruth_GradmagSynCar.txt'
    dataset_test = 'groundtruth_GradmagReal.txt'
    # directory = '/home/tarekfourati/pfa/RecurrentBIM-PoseNet/islam/'
    # dataset_train = 'met'
    # dataset_test = 'met2'
    training_generator = DataGenerator(
        directory=directory,
        file_name=dataset_train,
        window=window
    )
    test_generator = DataGenerator(
        directory=directory,
        file_name=dataset_test,
        window=window,
        mean=training_generator.mean
    )

    # Setup checkpointing for keeping the best results
    # checkpointer = ModelCheckpoint(filepath="today_batch25_LR0001_beta_600_brforgradmag_dropout.h5", verbose=1, save_best_only=True, save_weights_only=True)
    checkpointer = ModelCheckpoint(
        filepath='window' + str(window) + 'batch' + str(batch_size) + 'LR' + str(learningRate) + 'beta' + str(
            beta) + 'LSTM' + str(LSTM_size) + 'Dropout' + str(drop1) + str(drop2) + 'checkpoint.h5', verbose=1,
        save_best_only=True, save_weights_only=True, monitor='val_cls3_fc_pose_xyz_new_validation_error_x', mode='min')
    checkpointer2 = ModelCheckpoint(
        filepath='window' + str(window) + 'batch' + str(batch_size) + 'LR' + str(learningRate) + 'beta' + str(
            beta) + 'LSTM' + str(LSTM_size) + 'Dropout' + str(drop1) + str(drop2) + "epoch{epoch:03d}"+'.h5', verbose=1,
        save_weights_only=True,save_freq=20 )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    # creating history object to train and record the log of the process
    history = model.fit(
        training_generator,
        validation_data=test_generator,
        # add in callbacks
        # TestCallback([X_test_shuffle, y_test_x, y_test_q])
        callbacks=[checkpointer, tensorboard_callback, checkpointer2],
        # use_multiprocessing=True,
        # workers=2,
        epochs=100,
        initial_epoch=current_epoch

    )
    # todo

    #    history_dict = history.history
    #    print history_dict.keys()
    #    print history.history['val_cls3_fc_pose_xyz_new_acc']

    # Store the loss with each iteration
    with open('window' + str(window) + 'batch' + str(batch_size) + 'LR' + str(learningRate) + 'beta' + str(
            beta) + 'LSTM' + str(LSTM_size) + 'Dropout' + str(drop1) + str(drop2) + '.csv', "a+") as f:
        for ii in range(len(history.history['loss'])):
            f.write('{},{}\n'.format(str(history.history['loss'][ii]), str(history.history['val_loss'][ii])))

    # save the final model
    model.save_weights('window' + str(window) + 'batch' + str(batch_size) + 'LR' + str(learningRate) + 'beta' + str(
        beta) + 'LSTM' + str(LSTM_size) + 'Dropout' + str(drop1) + str(drop2) + '_weight.h5')

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.plot(history.history['mse'])
    # plt.plot(history.history['val_mse'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation', 'mse', 'val_mse'])
    # plt.gcf()
    # plt.savefig("images/metrics")
    #
    # plt.show()
