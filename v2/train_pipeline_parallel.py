import numpy as np
import multiprocessing
from queue import Queue

forward_queues = {
    'conv': Queue(),
    'relu': Queue(),
    'maxpool': Queue(),
    'flatten': Queue(),
    'linear': Queue(),
    'loss': Queue(),
    'output': Queue()
}

backward_queues = {
    'loss': Queue(),
    'linear': Queue(),
    'flatten': Queue(),
    'maxpool': Queue(),
    'relu': Queue(),
    'conv': Queue()
}

class PipelineServer():
    def __init__(self, model):
        self.model = model
        self.losses = []


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


class PipelineWorkerOne(multiprocessing.Process):
    def __init__(self, stage, is_forward, server, num_batches, all_y_labels, learning_rate, momentum_coeff):
        super().__init__()
        self.is_forward = is_forward
        self.stage = stage
        self.num_batches = num_batches
        self.all_y_labels = all_y_labels
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.server = server
        print(self.stage)

    def process_forwards(self):
        for i in range(self.num_batches):
            if self.stage == 'conv':
                inputs = forward_queues['conv'].get()
                print("1")
                conv_out = self.server.model.conv.forward(inputs)
                forward_queues['relu'].put(conv_out)
            elif self.stage == 'relu':
                conv_out = forward_queues['relu'].get()
                print("2")
                acti_out = self.server.model.relu.forward(conv_out)
                forward_queues['maxpool'].put(acti_out)
            elif self.stage == 'maxpool':
                acti_out = forward_queues['maxpool'].get()
                print("3")
                maxpool_out = self.server.model.maxpool.forward(acti_out)
                forward_queues['flatten'].put(maxpool_out)
            elif self.stage == 'flatten':
                maxpool_out = forward_queues['flatten'].get()
                print("4")
                flatten_out = self.server.model.flatten.forward(maxpool_out)
                forward_queues['linear'].put(flatten_out)
            elif self.stage == 'linear':
                flatten_out = forward_queues['linear'].get()
                linear_out = self.server.model.linear.forward(flatten_out)
                forward_queues['loss'].put(linear_out)
            elif self.stage == 'loss':
                linear_out = forward_queues['loss'].get()
                losses, preds = self.server.model.loss.forward(linear_out, self.all_y_labels[i], get_predictions=True)
                forward_queues['output'].put((losses, preds))
                backward_queues['loss'].put(True)

    def process_backwards(self):
        for _ in range(self.num_batches):
            if self.stage == 'loss':
                _ = backward_queues['loss'].get()
                loss_grad = self.server.model.loss.backward()
                backward_queues['linear'].put(loss_grad)
            elif self.stage == 'linear':
                loss_grad = backward_queues['linear'].get()
                linear_grad = self.server.model.linear.backward(loss_grad)
                backward_queues['flatten'].put(linear_grad)
                self.server.model.linear.update(self.learning_rate, self.momentum_coeff)
            elif self.stage == 'flatten':
                linear_grad = backward_queues['flatten'].get()
                flatten_grad = self.server.model.flatten.backward(linear_grad)
                backward_queues['maxpool'].put(flatten_grad)
            elif self.stage == 'maxpool':
                flatten_grad = backward_queues['maxpool'].get()
                maxpool_grad = self.server.model.maxpool.backward(flatten_grad)
                backward_queues['relu'].put(maxpool_grad)
            elif self.stage == 'relu':
                maxpool_grad = backward_queues['relu'].get()
                relu_grad = self.server.model.relu.backward(maxpool_grad)
                backward_queues['conv'].put(relu_grad)
            elif self.stage == 'conv':
                relu_grad = backward_queues['conv'].get()
                self.server.model.conv.backward(relu_grad)
                self.server.model.conv.update(self.learning_rate, self.momentum_coeff)

    def run(self):
        if self.is_forward:
            self.process_forwards()
        else:
            self.process_backwards()

def train_epoch_pipeline_parallel(model, batch_size, learning_rate, momentum_coeff, trainX, trainY, pureTrainY, testX, testY, pureTestY):
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
    yBatches = []

    # server.reset_losses()
    for i in range(num_batches):
        start = i*third_batch
        end = (i+1)*third_batch
        currX = np.concatenate([trainX[start:end], trainX[start + third_images:end + third_images], trainX[start + 2*third_images:end + 2*third_images]])
        currY = np.concatenate([trainY[start:end], trainY[start + third_images:end + third_images], trainY[start + 2*third_images:end + 2*third_images]])

        forward_queues['conv'].put(currX)
        yBatches.append(currY)

    workers = []
    for stage in model.get_stages():
        w1 = PipelineWorkerOne(stage, True, model, num_batches, yBatches, learning_rate, momentum_coeff)
        w2 = PipelineWorkerOne(stage, False, model, num_batches, yBatches, learning_rate, momentum_coeff)
        workers.append(w1)
        workers.append(w2)

    for worker in workers:
        worker.start()

    for _ in range(num_batches):
        loss, _ = forward_queues['output'].get()
        losses.append(loss)

    for worker in workers:
        worker.join()

    train_loss = np.sum(losses)/num_images
    _, train_pred, _ = model.forward(trainX, trainY)
    train_accu = np.count_nonzero(train_pred == pureTrainY)/num_images
    test_loss, test_pred, _ = model.forward(testX, testY)
    test_loss = test_loss/num_test
    test_accu = np.count_nonzero(test_pred == pureTestY)/num_test
    return train_loss, train_accu, test_loss, test_accu
