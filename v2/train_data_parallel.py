import numpy as np
from conv1d_data_parallel import ConvNetOneDataParallel
from conv2d_data_parallel import ConvNetTwoDataParallel
import multiprocessing

class ParamServer:
    def __init__(self, model, learning_rate, momentum_coeff):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.losses = []
        self.lock = multiprocessing.Lock()

    def get_weights(self):
        with self.lock:
            return self.model.get_weights()

    def update_gradients(self, loss, outputs, N):
        with self.lock:
            self.losses.append(loss)
            self.model.update(self.learning_rate, self.momentum_coeff, outputs, N)

    def reset_losses(self):
        self.losses = []

    def get_metrics(self, trainX, trainY, pureTrainY, testX, testY, pureTestY, num_images, num_test):
        train_loss = np.sum(self.losses)/num_images
        _, train_pred, _ = self.model.forward(trainX, trainY)
        train_accu = np.count_nonzero(train_pred == pureTrainY)/num_images
        test_loss, test_pred, _ = self.model.forward(testX, testY)
        test_loss = test_loss/num_test
        test_accu = np.count_nonzero(test_pred == pureTestY)/num_test
        return train_loss, train_accu, test_loss, test_accu

class ParamWorker(multiprocessing.Process):
    def __init__(self, numConv, param_server, data, labels):
        super().__init__()
        if numConv == 1:
            self.model = ConvNetOneDataParallel(out_dim=4, input_shape=(3,64,64), filter_shape=(1,5,5))
        else:
            self.model = ConvNetTwoDataParallel(out_dim=4, input_shape=(3,64,64), filter_shape=(1,5,5))
        self.param_server = param_server
        self.data = data
        self.labels = labels

    def compute_gradients(self):
        N, _, _,  _ = np.shape(self.data)
        (curr_loss, _, parallel_updates) = self.model.forward(self.data, self.labels)
        outputs = self.model.backward(parallel_updates)
        return N, curr_loss, outputs

    def run(self):
        new_weights = self.param_server.get_weights()
        self.model.override_weights(new_weights)
        N, curr_loss, outputs = self.compute_gradients()
        self.param_server.update_gradients(curr_loss, outputs, N)
    

def train_epoch_data_parallel(param_server, num_conv, batch_size, trainX, trainY, pureTrainY, testX, testY, pureTestY):
    num_images = np.shape(trainX)[0]
    num_test = np.shape(testX)[0]
    permut = np.random.choice(num_images, num_images, replace=False)
    trainX = trainX[permut]
    trainY = trainY[permut]
    pureTrainY = pureTrainY[permut]
    num_batches = int(np.ceil(num_images/batch_size))
    third_batch = int(batch_size/3)
    third_images = int(num_images/3)
    
    param_server.reset_losses()
    param_workers = []
    for i in range(num_batches):
        start = i*third_batch
        end = (i+1)*third_batch
        currX = np.concatenate([trainX[start:end], trainX[start + third_images:end + third_images], trainX[start + 2*third_images:end + 2*third_images]])
        currY = np.concatenate([trainY[start:end], trainY[start + third_images:end + third_images], trainY[start + 2*third_images:end + 2*third_images]])
        curr_worker = ParamWorker(num_conv, param_server, currX, currY)
        param_workers.append(curr_worker)
    
    for worker in param_workers:
        worker.start()
    
    for worker in param_workers:
        worker.join()
    

    return param_server.get_metrics(trainX, trainY, pureTrainY, testX, testY, pureTestY, num_images, num_test)
    