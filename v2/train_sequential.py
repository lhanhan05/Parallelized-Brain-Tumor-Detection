import numpy as np

def train_epoch_sequential(model, batch_size, learning_rate, momentum_coeff, trainX, trainY, pureTrainY, testX, testY, pureTestY):
    num_images = np.shape(trainX)[0]
    num_test = np.shape(testX)[0]
    permut = np.random.choice(num_images, num_images, replace=False)
    trainX = trainX[permut]
    trainY = trainY[permut]
    pureTrainY = pureTrainY[permut]
    num_batches = int(np.ceil(num_images/batch_size))
    losses = []
    third_batch = int(batch_size/3)
    third_images = int(num_images/3)
    for i in range(num_batches):
        start = i*third_batch
        end = (i+1)*third_batch
        currX = np.concatenate([trainX[start:end], trainX[start + third_images:end + third_images], trainX[start + 2*third_images:end + 2*third_images]])
        currY = np.concatenate([trainY[start:end], trainY[start + third_images:end + third_images], trainY[start + 2*third_images:end + 2*third_images]])
        curr_loss,_ = model.forward(currX, currY)
        losses.append(curr_loss)
        model.backward()
        model.update(learning_rate, momentum_coeff)
    train_loss = np.sum(losses)/num_images
    _, train_pred = model.forward(trainX, trainY)
    train_accu = np.count_nonzero(train_pred == pureTrainY)/num_images
    test_loss, test_pred = model.forward(testX, testY)
    test_loss = test_loss/num_test
    test_accu = np.count_nonzero(test_pred == pureTestY)/num_test
    return train_loss, train_accu, test_loss, test_accu