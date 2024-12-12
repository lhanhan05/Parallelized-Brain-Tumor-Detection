import numpy as np
import os
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from conv1d_sequential import ConvNetOneSequential
from conv1d_data_parallel import ConvNetOneDataParallel
from train_sequential import train_epoch_sequential
from train_data_parallel import train_epoch_data_parallel
from conv2d_sequential import ConvNetTwoSequential


def img_to_matrix(path):
    img = Image.open(path)
    small_img = img.resize((64,64), Image.Resampling.LANCZOS)
    arr = np.array(small_img)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    arr = arr.transpose((2, 0, 1))
    return arr

def create_data(classes, test_size = 0.2):
    X = []
    Y = []
    dir = "../Data"
    
    for i in range(4):
        curr_class = classes[i]
        curr_path = os.path.join(dir, curr_class)
        for file in sorted(os.listdir(curr_path)):
            file_path = os.path.join(curr_path, file)
            img_arr = img_to_matrix(file_path)
            X.append(img_arr)
            Y.append(i)
            
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = np.array(X)
    Y = np.array(Y)

    test_size = int(len(X) * test_size)
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]
    trainX = X[train_indices]
    testX = X[test_indices]
    trainY = Y[train_indices]
    testY = Y[test_indices]
    return trainX, testX, trainY, testY
 
def one_hot_encode(labels):
    one_hot_labels = np.array([[i==label for i in range(4)] for label in labels], np.int32)
    return one_hot_labels

def prep_image_data(train_images, train_labels, test_images, test_labels):
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    train_images = (train_images - 127.5) / 127.5
    test_images = (test_images - 127.5) / 127.5

    return train_images, train_labels, test_images, test_labels


def train_model(model, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, trainX, trainY, pureTrainY, testX, testY, pureTestY, is_data_parallel):
    train_losses = []
    train_accus = []
    test_losses = []
    test_accus = []
    total_times = []
    idxs = []
    start_time = time.time()
    for i in tqdm(range(EPOCHS)):
        if is_data_parallel:
            train_loss, train_accu, test_loss, test_accu = train_epoch_data_parallel(model, BATCH_SIZE, LEARNING_RATE, MOMENTUM, trainX, trainY, pureTrainY, testX, testY, pureTestY)
        else:
            train_loss, train_accu, test_loss, test_accu = train_epoch_sequential(model, BATCH_SIZE, LEARNING_RATE, MOMENTUM, trainX, trainY, pureTrainY, testX, testY, pureTestY)
        curr_time = time.time()-start_time
        idxs.append(i)
        train_losses.append(train_loss)
        train_accus.append(train_accu)
        test_losses.append(test_loss)
        test_accus.append(test_accu)
        total_times.append(curr_time)
        print("Epoch {} done: {}, {}, {}, {}, {}s".format(i, train_loss, train_accu, test_loss, test_accu, curr_time))

    
    print("Best Training Loss: {}".format(np.min(train_losses)))
    print("Best Test Loss: {}".format(np.min(test_losses)))
    print("Best Train Accuracy: {}".format(np.max(train_accus)))
    print("Best Test Accuracy : {}".format(np.max(test_accus)))
    print("Elapsed Time: {}s".format(np.max(total_times)))
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(idxs, train_losses, label='train loss')
    axs[0].plot(idxs, test_losses, label='test loss')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')
    
    axs[1].plot(idxs, train_accus, label='train accuracy')
    axs[1].plot(idxs, test_accus, label='test accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='upper left')
    

    plt.savefig('1d_seq.png')
    # plt.savefig('2d_seq.png')

    # plt.savefig('1d_data.png')
    # plt.savefig('2d_data.png')
    print("Total Times per Epoch:", total_times)


if __name__ == '__main__':
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MOMENTUM = 0.95
    EPOCHS = 50
    classes = ['glioma', 'meningioma', 'normal', 'pituitary']
    trainX, testX, pureTrainY, pureTestY = create_data(classes, test_size=0.2)
    trainX, trainY, testX, testY = prep_image_data(trainX, pureTrainY, testX, pureTestY)

    # RUNNING SEQUENTIAL
    # modelSequential = ConvNetOneSequential(out_dim=4, input_shape=(3,64,64), filter_shape=(1,5,5))
    # modelSequential = ConvNetTwoSequential(out_dim=4, input_shape=(3,64,64), filter_shape=(1,5,5))
    # train_model(modelSequential, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, trainX, trainY, pureTrainY, testX, testY, pureTestY, False)

    # RUNNING DATA PARALLELISM
    modelDataParallel = ConvNetOneDataParallel(out_dim=4, input_shape=(3,64,64), filter_shape=(1,5,5))
    train_model(modelDataParallel, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, trainX, trainY, pureTrainY, testX, testY, pureTestY, True)
