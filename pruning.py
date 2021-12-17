import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import h5py
import numpy as np


clean_valid = "data/cl/valid.h5"
clean_test = "data/cl/test.h5"
poisoned_valid = "data/bd/bd_valid.h5"
poisoned_test = "data/bd/bd_test.h5"

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data


def evaluate_bd_net():
    cl_x_valid, cl_y_valid = data_loader(clean_valid)
    bd_x_valid, bd_y_valid = data_loader(poisoned_valid)

    bd_model = keras.models.load_model("models/bd_net.h5")
    bd_model.load_weights("models/bd_weights.h5")
    bd_model.summary()
    '''
    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input (InputLayer)              [(None, 55, 47, 3)]  0
    __________________________________________________________________________________________________
    conv_1 (Conv2D)                 (None, 52, 44, 20)   980         input[0][0]
    __________________________________________________________________________________________________
    pool_1 (MaxPooling2D)           (None, 26, 22, 20)   0           conv_1[0][0]
    __________________________________________________________________________________________________
    conv_2 (Conv2D)                 (None, 24, 20, 40)   7240        pool_1[0][0]
    __________________________________________________________________________________________________
    pool_2 (MaxPooling2D)           (None, 12, 10, 40)   0           conv_2[0][0]
    __________________________________________________________________________________________________
    conv_3 (Conv2D)                 (None, 10, 8, 60)    21660       pool_2[0][0]
    __________________________________________________________________________________________________
    pool_3 (MaxPooling2D)           (None, 5, 4, 60)     0           conv_3[0][0]
    __________________________________________________________________________________________________
    conv_4 (Conv2D)                 (None, 4, 3, 80)     19280       pool_3[0][0]
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 1200)         0           pool_3[0][0]
    __________________________________________________________________________________________________
    flatten_2 (Flatten)             (None, 960)          0           conv_4[0][0]
    __________________________________________________________________________________________________
    fc_1 (Dense)                    (None, 160)          192160      flatten_1[0][0]
    __________________________________________________________________________________________________
    fc_2 (Dense)                    (None, 160)          153760      flatten_2[0][0]
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 160)          0           fc_1[0][0]
                                                                    fc_2[0][0]
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 160)          0           add_1[0][0]
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1283)         206563      activation_1[0][0]
    ==================================================================================================
    '''
    cl_label_p = np.argmax(bd_model.predict(cl_x_valid), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_valid))*100
    print('Clean Classification accuracy:', clean_accuracy)
    
    bd_label_p = np.argmax(bd_model.predict(bd_x_valid), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_valid))*100
    print('Attack Success Rate:', asr)

def evaluate_pruning():
    cl_x_valid, cl_y_valid = data_loader(clean_valid)
    bd_x_valid, bd_y_valid = data_loader(poisoned_valid)

    bd_model = keras.models.load_model("models/bd_net.h5")
    bd_model.load_weights("models/bd_weights.h5")
    #bd_model.summary()

    # evaluate the original model clean images accuracy
    cl_label_p = np.argmax(bd_model.predict(cl_x_valid), axis=1)
    clean_accuracy_unpruned = np.mean(np.equal(cl_label_p, cl_y_valid))*100
    print('Clean Classification accuracy for unpruned model:', clean_accuracy_unpruned)

    # evaluate the original model poisoned images accuracy
    bd_label_p = np.argmax(bd_model.predict(bd_x_valid), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_valid))*100
    print('Attack Success Rate for unpruned model:', asr)


    pool3_layer = bd_model.get_layer('pool_3')
    keras_function = keras.backend.function([bd_model.input], [pool3_layer.output])
    # calculate activations over the whole clean validation set.
    target_layer_output = keras_function([cl_x_valid])
    overall_activations = np.mean(np.array(target_layer_output), axis=(0, 1, 2, 3))
    
    # sort the activations in decreasinig order to pick the sequence to prune
    #pruning_order_sorted = np.argsort(overall_activations)[::-1] 
    pruning_order_sorted = np.argsort(overall_activations)
    '''[ 0 26 27 30 31 33 34 36 37 38 25 39 41 44 45 47 48 49 50 53 55 40 24 59
      9  2 12 13 17 14 15 23  6 51 32 22 21 20 19 43 58  3 42  1 29 16 56 46
      5  8 11 54 10 28 35 18  4  7 52 57]
    '''
    channel_size = len(pruning_order_sorted) # 60

    conv3weight = bd_model.get_layer('conv_3').get_weights()[0]  # [3, 3, 40, 60]
    conv3bias = bd_model.get_layer('conv_3').get_weights()[1] # [60]
    print(conv3weight.shape)
    print(conv3bias.shape)
    #print(conv3bias)

    save_points_X = [2, 4, 10]
    save_id = 0

    clean_acc_log = open("cl_acc_log_ascending.txt", "w")
    poison_acc_log = open("ps_acc_log_ascending.txt", "w")

    for pruned_num, pruned_target_channel_id in enumerate(pruning_order_sorted):
        print("Pruning the ", pruned_num+1, "th channel with channel number ", pruned_target_channel_id)
        conv3weight[:, :, :, pruned_target_channel_id] = 0
        conv3bias[pruned_target_channel_id] = 0
        #print(conv3bias)
        bd_model.get_layer('conv_3').set_weights([conv3weight, conv3bias])

        # evaluate the model
        cl_label_p = np.argmax(bd_model.predict(cl_x_valid), axis=1)
        clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_valid))*100
        print('Clean Classification accuracy:', clean_accuracy)
        clean_acc_log.write(str(clean_accuracy) + "\n")
        clean_acc_log.flush()

        bd_label_p = np.argmax(bd_model.predict(bd_x_valid), axis=1)
        asr = np.mean(np.equal(bd_label_p, bd_y_valid))*100
        print('Attack Success Rate:', asr)
        poison_acc_log.write(str(poison_acc_log) + "\n")
        poison_acc_log.flush()

        # save the weights if the accuracy drops below X % = {2%, 4%, 10%}
        if save_id < 3 and clean_accuracy_unpruned - clean_accuracy > save_points_X[save_id]:
            bd_model.save("models/pruned_" + str(save_points_X[save_id]) + "_ascending.h5")
            save_id += 1

if __name__ == '__main__':
    evaluate_pruning()
    