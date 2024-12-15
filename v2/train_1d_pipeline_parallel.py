import numpy as np
import multiprocessing
import time
from queue import Queue

global_idle_times = [] # Global list to store idle times

class PipelineServerOne():
    def __init__(self, model):
        self.model = model
        self.stages = ['loss', 'linear', 'flatten', 'maxpool', 'relu', 'conv']
        self.losses = []
        self.locks = {
            'loss': multiprocessing.Lock(),
            'linear': multiprocessing.Lock(),
            'flatten': multiprocessing.Lock(),
            'maxpool': multiprocessing.Lock(),
            'relu': multiprocessing.Lock(),
            'conv': multiprocessing.Lock(),
        }
        self.idle_times =  multiprocessing.Manager.list() # Track idle times


    def forward(self, stage, input):
        with self.locks[stage]:
            if stage == 'conv':
                return self.model.conv.forward(input)
            elif stage == 'relu':
                return self.model.relu.forward(input)
            elif stage == 'maxpool':
                return self.model.maxpool.forward(input)
            elif stage == 'flatten':
                return self.model.flatten.forward(input)
            elif stage == 'linear':
                return self.model.linear.forward(input)
            elif stage == 'loss':
                linear_out, labels = input
                return self.model.loss.forward(linear_out, labels, True)
        
    def backward(self, stage, input):
        with self.locks[stage]:
            if stage == 'conv':
                return self.model.conv.backward(input)
            elif stage == 'relu':
                return self.model.relu.backward(input)
            elif stage == 'maxpool':
                return self.model.maxpool.backward(input)
            elif stage == 'flatten':
                return self.model.flatten.backward(input)
            elif stage == 'linear':
                return self.model.linear.backward(input)
            elif stage == 'loss':
                return self.model.loss.backward()
        
    def update(self, stage, learning_rate, momentum_coeff):
        if stage == 'linear':
            self.model.linear.update(learning_rate, momentum_coeff)
        elif stage == 'conv':
            self.model.conv.update(learning_rate, momentum_coeff)
        
    def get_stages(self):
        return self.stages
    
    def get_weights(self):
        return self.model.get_weights()
        

    def reset_losses(self):
        self.losses = []

    def get_metrics(self, trainX, trainY, pureTrainY, testX, testY, pureTestY, num_images, num_test, total_idle_time):
        train_loss = np.sum(self.losses)/num_images
        train_loss, train_pred = self.model.forward(trainX, trainY)
        train_accu = np.count_nonzero(train_pred == pureTrainY)/num_images
        test_loss, test_pred = self.model.forward(testX, testY)
        test_loss = test_loss/num_test
        train_loss = train_loss/num_images
        test_accu = np.count_nonzero(test_pred == pureTestY)/num_test
        return train_loss, train_accu, test_loss, test_accu, total_idle_time

    def get_idle_times(self):
        return self.idle_times
    
class PipelineWorkerOne(multiprocessing.Process):
    def __init__(self, stage, is_forward, server, num_batches, all_y_labels, learning_rate, momentum_coeff, forward_queues, backward_queues):
        super().__init__()
        self.is_forward = is_forward
        self.stage = stage
        self.num_batches = num_batches
        self.all_y_labels = all_y_labels
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.server = server
        self.forward_queues = forward_queues
        self.backward_queues = backward_queues
        self.idle_time = 0.0  # To track idle time for this worker


    def process_forwards(self):
        start_time = time.time()
        for i in range(self.num_batches):
            start_idle = time.time()  # Start measuring idle time

            if self.stage == 'conv':
                inputs = self.forward_queues['conv'].get()
                self.idle_time += time.time() - start_idle
                conv_out = self.server.forward('conv', inputs)
                self.forward_queues['relu'].put(conv_out)
            elif self.stage == 'relu':
                conv_out = self.forward_queues['relu'].get()
                self.idle_time += time.time() - start_idle
                acti_out = self.server.forward('relu', conv_out)
                self.forward_queues['maxpool'].put(acti_out)
            elif self.stage == 'maxpool':
                acti_out = self.forward_queues['maxpool'].get()
                self.idle_time += time.time() - start_idle
                maxpool_out = self.server.forward('maxpool', acti_out)
                self.forward_queues['flatten'].put(maxpool_out)
            elif self.stage == 'flatten':
                maxpool_out = self.forward_queues['flatten'].get()
                self.idle_time += time.time() - start_idle
                flatten_out = self.server.forward('flatten', maxpool_out)
                self.forward_queues['linear'].put(flatten_out)
            elif self.stage == 'linear':
                flatten_out = self.forward_queues['linear'].get()
                self.idle_time += time.time() - start_idle
                linear_out = self.server.forward('linear', flatten_out)
                self.forward_queues['loss'].put(linear_out)
            elif self.stage == 'loss':
                linear_out = self.forward_queues['loss'].get()
                self.idle_time += time.time() - start_idle
                losses, preds = self.server.forward('loss', (linear_out, self.all_y_labels[i]))
                self.backward_queues['loss'].put(True)

    def process_backwards(self):
        for _ in range(self.num_batches):
            start_idle = time.time()  # Start measuring idle time

            if self.stage == 'loss':
                _ = self.backward_queues['loss'].get()
                self.idle_time += time.time() - start_idle
                loss_grad = self.server.backward('loss', None)
                self.backward_queues['linear'].put(loss_grad)
            elif self.stage == 'linear':
                loss_grad = self.backward_queues['linear'].get()
                self.idle_time += time.time() - start_idle
                linear_grad = self.server.backward('linear', loss_grad)
                self.backward_queues['flatten'].put(linear_grad)
                self.server.update('linear', self.learning_rate, self.momentum_coeff)
            elif self.stage == 'flatten':
                linear_grad = self.backward_queues['flatten'].get()
                self.idle_time += time.time() - start_idle
                flatten_grad = self.server.backward('flatten', linear_grad[2])
                self.backward_queues['maxpool'].put(flatten_grad)
            elif self.stage == 'maxpool':
                flatten_grad = self.backward_queues['maxpool'].get()
                self.idle_time += time.time() - start_idle
                maxpool_grad = self.server.backward('maxpool', flatten_grad)
                self.backward_queues['relu'].put(maxpool_grad)
            elif self.stage == 'relu':
                maxpool_grad = self.backward_queues['relu'].get()
                self.idle_time += time.time() - start_idle
                relu_grad = self.server.backward('relu', maxpool_grad)
                self.backward_queues['conv'].put(relu_grad)
            elif self.stage == 'conv':
                relu_grad = self.backward_queues['conv'].get()
                self.idle_time += time.time() - start_idle
                self.server.backward('conv', relu_grad)
                self.server.update('conv', self.learning_rate, self.momentum_coeff)            

    def run(self):
        if self.is_forward:
            self.process_forwards()
        else:
            self.process_backwards()

        self.server.idle_times.append(self.idle_time)

def train_epoch_pipeline_parallel_1d(server, batch_size, learning_rate, momentum_coeff, trainX, trainY, pureTrainY, testX, testY, pureTestY):
    num_images = np.shape(trainX)[0]
    num_test = np.shape(testX)[0]
    permut = np.random.choice(num_images, num_images, replace=False)
    trainX = trainX[permut]
    trainY = trainY[permut]
    pureTrainY = pureTrainY[permut]
    num_batches = int(np.ceil(num_images/batch_size))
    third_batch = int(batch_size/3)
    third_images = int(num_images/3)
    yBatches = []
    xBatches = []

    with multiprocessing.Manager() as manager:
        forward_queues = {
            'conv': manager.Queue(),
            'relu': manager.Queue(),
            'maxpool': manager.Queue(),
            'flatten': manager.Queue(),
            'linear': manager.Queue(),
            'loss': manager.Queue(),
            'output': manager.Queue()
        }
        backward_queues = {
            'loss': manager.Queue(),
            'linear': manager.Queue(),
            'flatten': manager.Queue(),
            'maxpool': manager.Queue(),
            'relu': manager.Queue(),
            'conv': manager.Queue()
        }
        for i in range(num_batches):
            start = i * third_batch
            end = (i + 1) * third_batch
            currX = np.concatenate([trainX[start:end], trainX[start + third_images:end + third_images], trainX[start + 2 * third_images:end + 2 * third_images]])
            currY = np.concatenate([trainY[start:end], trainY[start + third_images:end + third_images], trainY[start + 2 * third_images:end + 2 * third_images]])

            xBatches.append(currX)
            forward_queues['conv'].put(currX)
            yBatches.append(currY)

        workers = []
        for stage in server.get_stages():
            w1 = PipelineWorkerOne(stage, True, server, num_batches, yBatches, learning_rate, momentum_coeff, forward_queues, backward_queues)
            w2 = PipelineWorkerOne(stage, False, server, num_batches, yBatches, learning_rate, momentum_coeff, forward_queues, backward_queues)
            workers.append(w1)
            workers.append(w2)

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        total_idle_time = sum(server.get_idle_times())
        return server.get_metrics(trainX, trainY, pureTrainY, testX, testY, pureTestY, num_images, num_test, total_idle_time)
