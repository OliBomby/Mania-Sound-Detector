import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Generate_Data3 as Generate_Data
import keras
1

save_path = Generate_Data.save_path
save_rate = 50000

audio_size = Generate_Data.audio_size

batch_size = 10
divisor = 4;
time_interval = 16;

def read_npz(fn):
    with np.load(fn) as data:
        wav_data = data["wav"];
        wav_data = np.swapaxes(wav_data, 2, 3);
        train_data = wav_data;
        div_source = data["lst"][:, 0];
        div_source2 = data["lst"][:, 12:15];
        div_data = np.concatenate([np.array([[int(k%4==0), int(k%4==1), int(k%4==2), int(k%4==3)] for k in div_source]), div_source2], axis=1);
        lst_data = data["lst"][:, 2:10];
        # Change the 0/1 data to -1/1 to use tanh instead of softmax in the NN.
        # Somehow tanh works much better than softmax, even if it is a linear combination. Maybe because it is alchemy!
        lst_data = 2 * lst_data - 1;
        train_labels = lst_data;
    return train_data, div_data, train_labels;

def read_npz_list():
    npz_list = [];
    for file in os.listdir(root):
        if file.endswith(".npz"):
            npz_list.append(os.path.join(root, file));
    # reutnr npz_lsit;
    return npz_list;

def prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    # Filter out slider ends from the training set, since we cannot reliably decide if a slider end is on a note.
    # Another way is to set 0.5 for is_note value, but that will break the validation algorithm.
    # Also remove the IS_SLIDER_END, IS_SPINNER_END columns which are left to be zeros.

    # Before: NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_SLIDER_END, IS_SPINNER_END, SLIDING, SPINNING
    #            0,         1,         2,          3,             4,              5,       6,        7
    # After:  NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, SLIDING, SPINNING
    #            0,         1,         2,          3,       4,        5

    non_object_end_indices = [i for i,k in enumerate(train_labels_unfiltered) if k[4] == -1 and k[5] == -1];
    train_data = train_data_unfiltered[non_object_end_indices];
    div_data = div_data_unfiltered[non_object_end_indices];
    train_labels = train_labels_unfiltered[non_object_end_indices][:, [0, 1, 2, 3, 6, 7]];
    
    # should be (X, 7, 32, 2) and (X, 6) in default sampling settings
    # (X, fft_window_type, freq_point, magnitude/phase)
    return train_data, div_data, train_labels;

def preprocess_npzs(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    train_data, div_data, train_labels = train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered;
    # In this version, the train data is already normalized, no need to do it again here
#     mean = train_data.mean(axisprefilter_=0)
#     std = train_data.std(axis=0)
#     train_data = (train_data - np.tile(mean, (train_data.shape[0], 1,1,1))) / np.tile(std, (train_data.shape[0], 1,1,1))
    
    # Make time intervals from training data
    if train_data.shape[0]%time_interval > 0:
        train_data = train_data[:-(train_data.shape[0]%time_interval)];
        div_data = div_data[:-(div_data.shape[0]%time_interval)];
        train_labels = train_labels[:-(train_labels.shape[0]%time_interval)];
    train_data2 = np.reshape(train_data, (-1, time_interval, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    div_data2 = np.reshape(div_data, (-1, time_interval, div_data.shape[1]))
    train_labels2 = np.reshape(train_labels, (-1, time_interval, train_labels.shape[1]))
    return train_data2, div_data2, train_labels2;

def get_data_shape():
    for file in os.listdir(root):
        if file.endswith(".npz"):
            train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered = read_npz(os.path.join(root, file));
            train_data, div_data, train_labels = prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered);
            # should be (X, 7, 32, 2) and (X, 6) in default sampling settings
            # (X, fft_window_type, freq_point, magnitude/phase)
            # X = 76255
            # print(train_data.shape, train_labels.shape);
            if train_data.shape[0] == 0:
                continue;
            return train_data.shape, div_data.shape, train_labels.shape;
    print("cannot find npz!! using default shape");
    return (-1, 7, 32, 2), (-1, 4), (-1, 6);

def read_some_npzs_and_preprocess():
    x_data, y_data, tick_data = Generate_Data.load_training_data()

    x_data = np.swapaxes(x_data, 2, 3);
    div_data = np.array([[int(k%4==0), int(k%4==1), int(k%4==2), int(k%4==3)] for k in tick_data])
    
    train_data_unfiltered = x_data;
    div_data_unfiltered = div_data;
    train_labels_unfiltered = y_data;

    train_data2, div_data2, train_labels2 = preprocess_npzs(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered);
    return train_data2, div_data2, train_labels2;

def train_test_split(train_data2, div_data2, train_labels2, test_split_count=10):
    new_train_data = train_data2[:-test_split_count];
    new_div_data = div_data2[:-test_split_count];
    new_train_labels = train_labels2[:-test_split_count];
    test_data = train_data2[-test_split_count:];
    test_div_data = div_data2[-test_split_count:];
    test_labels = train_labels2[-test_split_count:];
    return (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels);




train_shape, div_shape, label_shape = (-1, 7, 32, 2), (-1, 4), (-1, 1)


from keras.models import Model;
def build_model():
    model1 = keras.Sequential([
        keras.layers.TimeDistributed(keras.layers.Conv2D(16, (2, 2),
                           data_format='channels_last'),
                           input_shape=(time_interval, train_shape[1], train_shape[2], train_shape[3])),
        keras.layers.TimeDistributed(keras.layers.MaxPool2D((1, 2),
                           data_format='channels_last')),
        keras.layers.TimeDistributed(keras.layers.Activation(activation=tf.nn.relu)),
        keras.layers.TimeDistributed(keras.layers.Dropout(0.3)),
        keras.layers.TimeDistributed(keras.layers.Conv2D(16, (2, 3),
                           data_format='channels_last')),
        keras.layers.TimeDistributed(keras.layers.MaxPool2D((1, 2),
                           data_format='channels_last')),
        keras.layers.TimeDistributed(keras.layers.Activation(activation=tf.nn.relu)),
        keras.layers.TimeDistributed(keras.layers.Dropout(0.3)),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.LSTM(64, activation=tf.nn.tanh, return_sequences=True)
    ])
    
    input2 = keras.layers.InputLayer(input_shape=(time_interval, div_shape[1]));
    
    conc = keras.layers.concatenate([model1.output, input2.output]);
    dense1 = keras.layers.Dense(71, activation=tf.nn.tanh)(conc);
    dense2 = keras.layers.Dense(71, activation=tf.nn.relu)(dense1);
    dense3 = keras.layers.Dense(label_shape[1], activation=tf.nn.relu)(dense2);
    

    optimizer = keras.optimizers.RMSprop(lr=0.001);

    
    final_model = Model(inputs=[model1.input, input2.input], outputs=dense3);
    final_model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[keras.metrics.mae])
    return final_model

if __name__ == '__main__':

    model = build_model()
    model.summary()

    if input("Do you want to restore the model?(y/n): ") == "y":
        model = load_model('model3.h5')
        print("Model restored.")
            
    def plot_history(history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [Limitless]')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
               label='Train MAE')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
               label = 'Val MAE')
        plt.plot(history.epoch, np.array(history.history['loss']), 
               label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_loss']),
               label = 'Val Loss')
        plt.legend()
        plt.show()

    def plot(ll, predictions, labels):
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [Limitless]')
        plt.subplot(1, 3, 1)
        plt.cla()
        plt.plot(ll)

        plt.subplot(1, 3, 3)
        plt.cla()
        plt.plot(labels, color="red")
        plt.plot(predictions, color="green")

        plt.draw()
        plt.pause(0.0001)

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 100 == 0: print('')
            #print('.', end='')
            print(logs['loss'])

    early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=20)

    EPOCHS = 100000000


    ##train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess();
    ##
    ### Split some test data out
    ##(new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2);
    ##
    ##
    ##
    ### Store training stats
    ##history = model.fit([new_train_data, new_div_data], new_train_labels, epochs=EPOCHS,
    ##                    validation_split=0.2, verbose=0, #batch_size=10,
    ##                    callbacks=[early_stop, PrintDot()])
    ##
    ##plot_history(history)


    train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess();

    # Split some test data out
    (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2);


    plt.ion()
    plt.figure()
    plt.show()

    loss_list = []
    for epoch in range(EPOCHS):
        history = model.fit([new_train_data, new_div_data], new_train_labels, epochs=1,
                            validation_split=0.2, verbose=0, #batch_size=10,
                            callbacks=[])
        # Manually print the dot
        print('Epoch: %s Loss: %s' % (epoch, history.history['loss']))
        loss_list.append(history.history['loss'])

        index = np.random.randint(0,len(new_train_data))
        test_predictions = model.predict([new_train_data[index:index+1], new_div_data[index:index+1]]).reshape((-1, time_interval, label_shape[1]))

        flat_test_preds = test_predictions.reshape(-1, label_shape[1]);
        flat_test_labels = new_train_labels[index:index+1].reshape(-1, label_shape[1]);
        
        plot(loss_list, flat_test_preds, flat_test_labels)
        
        if epoch % 50 == 0 and epoch > 0:
            print("Saving model")
            model.save('model3.h5')
            print("Saved")


    [loss, mae] = model.evaluate([test_data, test_div_data], test_labels, verbose=0)
    print("\nTesting set Mean Abs Error: {}".format(mae))


    print("Saving model")
    model.save('model3.h5')
    print("Saved")

    plt.ioff()



