import numpy as np
import multiprocessing
import time  # To measure idle times
from queue import Queue


class PipelineServerTwo():
    def __init__(self, model):
        self.model = model
        self.stages = ['loss', 'linear', 'flatten', 'maxpool2', 'relu2', 'conv2', 'maxpool1', 'relu1', 'conv1']
        self.losses = []
        self.locks = {
            'loss': multiprocessing.Lock(),
            'linear': multiprocessing.Lock(),
            'flatten': multiprocessing.Lock(),
            'maxpool2': multiprocessing.Lock(),
            'relu2': multiprocessing.Lock(),
            'conv2': multiprocessing.Lock(),
            'maxpool1': multiprocessing.Lock(),
            'relu1': multiprocessing.Lock(),
            'conv1': multiprocessing.Lock(),
        }
        self.idle_times = multiprocessing.Manager().list()  # Track idle times



    def forward(self, stage, input):
        with self.locks[stage]:
            if stage == 'conv1':
                return self.model.conv1.forward(input)
            elif stage == 'relu1':
                return self.model.relu1.forward(input)
            elif stage == 'maxpool1':
                return self.model.maxpool1.forward(input)
            elif stage == 'conv2':
                return self.model.conv2.forward(input)
            elif stage == 'relu2':
                return self.model.relu2.forward(input)
            elif stage == 'maxpool2':
                return self.model.maxpool2.forward(input)
            elif stage == 'flatten':
                return self.model.flatten.forward(input)
            elif stage == 'linear':
                return self.model.linear.forward(input)
            elif stage == 'loss':
                linear_out, labels = input
                return self.model.loss.forward(linear_out, labels, True)
        
    def backward(self, stage, input):
        with self.locks[stage]:
            if stage == 'conv1':
                return self.model.conv1.backward(input)
            elif stage == 'relu1':
                return self.model.relu1.backward(input)
            elif stage == 'maxpool1':
                return self.model.maxpool1.backward(input)
            elif stage == 'conv2':
                return self.model.conv2.backward(input)
            elif stage == 'relu2':
                return self.model.relu2.backward(input)
            elif stage == 'maxpool2':
                return self.model.maxpool2.backward(input)
            elif stage == 'flatten':
                return self.model.flatten.backward(input)
            elif stage == 'linear':
                return self.model.linear.backward(input)
            elif stage == 'loss':
                return self.model.loss.backward()
        
    def update(self, stage, learning_rate, momentum_coeff):
        if stage == 'linear':
            self.model.linear.update(learning_rate, momentum_coeff)
        elif stage == 'conv1':
            self.model.conv1.update(learning_rate, momentum_coeff)
        elif stage == 'conv2':
            self.model.conv2.update(learning_rate, momentum_coeff)
        
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
    
class PipelineWorkerTwo(multiprocessing.Process):
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
        for i in range(self.num_batches):
            start_idle = time.time()  # Start measuring idle time

            if self.stage == 'conv1':
                inputs = self.forward_queues['conv1'].get()
                self.idle_time += time.time() - start_idle
                conv1_out = self.server.forward('conv1', inputs)
                self.forward_queues['relu1'].put(conv1_out)
            elif self.stage == 'relu1':
                conv1_out = self.forward_queues['relu1'].get()
                self.idle_time += time.time() - start_idle
                acti1_out = self.server.forward('relu1', conv1_out)
                self.forward_queues['maxpool1'].put(acti1_out)
            elif self.stage == 'maxpool1':
                acti1_out = self.forward_queues['maxpool1'].get()
                self.idle_time += time.time() - start_idle
                maxpool1_out = self.server.forward('maxpool1', acti1_out)
                self.forward_queues['conv2'].put(maxpool1_out)
            elif self.stage == 'conv2':
                inputs = self.forward_queues['conv2'].get()
                self.idle_time += time.time() - start_idle
                conv2_out = self.server.forward('conv2', inputs)
                self.forward_queues['relu2'].put(conv2_out)
            elif self.stage == 'relu2':
                conv2_out = self.forward_queues['relu2'].get()
                self.idle_time += time.time() - start_idle
                acti2_out = self.server.forward('relu2', conv2_out)
                self.forward_queues['maxpool2'].put(acti2_out)
            elif self.stage == 'maxpool2':
                acti2_out = self.forward_queues['maxpool2'].get()
                self.idle_time += time.time() - start_idle
                maxpool2_out = self.server.forward('maxpool2', acti2_out)
                self.forward_queues['flatten'].put(maxpool2_out)
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
            start_idle = time.time() 
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
                self.backward_queues['maxpool2'].put(flatten_grad)
            elif self.stage == 'maxpool2':
                flatten_grad = self.backward_queues['maxpool2'].get()
                self.idle_time += time.time() - start_idle
                maxpool_grad = self.server.backward('maxpool2', flatten_grad)
                self.backward_queues['relu2'].put(maxpool_grad)
            elif self.stage == 'relu2':
                maxpool2_grad = self.backward_queues['relu2'].get()
                self.idle_time += time.time() - start_idle
                relu2_grad = self.server.backward('relu2', maxpool2_grad)
                self.backward_queues['conv2'].put(relu2_grad)
            elif self.stage == 'conv2':
                relu2_grad = self.backward_queues['conv2'].get()
                self.idle_time += time.time() - start_idle
                conv2_grad = self.server.backward('conv2', relu2_grad)
                self.backward_queues['maxpool1'].put(conv2_grad)
                self.server.update('conv2', self.learning_rate, self.momentum_coeff)
            elif self.stage == 'maxpool1':
                conv2_grad = self.backward_queues['maxpool1'].get()
                self.idle_time += time.time() - start_idle
                maxpool1_grad = self.server.backward('maxpool1', conv2_grad[2])
                self.backward_queues['relu1'].put(maxpool1_grad)
            elif self.stage == 'relu1':
                maxpool1_grad = self.backward_queues['relu1'].get()
                self.idle_time += time.time() - start_idle
                relu1_grad = self.server.backward('relu1', maxpool1_grad)
                self.backward_queues['conv1'].put(relu1_grad)
            elif self.stage == 'conv1':
                relu1_grad = self.backward_queues['conv1'].get()
                self.idle_time += time.time() - start_idle
                conv1_grad = self.server.backward('conv1', relu1_grad)
                self.server.update('conv1', self.learning_rate, self.momentum_coeff)            

    def run(self):
        if self.is_forward:
            self.process_forwards()
        else:
            self.process_backwards()

        self.server.idle_times.append(self.idle_time)

def train_epoch_pipeline_parallel_2d(server, batch_size, learning_rate, momentum_coeff, trainX, trainY, pureTrainY, testX, testY, pureTestY):
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
            'conv1': manager.Queue(),
            'relu1': manager.Queue(),
            'maxpool1': manager.Queue(),
            'conv2': manager.Queue(),
            'relu2': manager.Queue(),
            'maxpool2': manager.Queue(),
            'flatten': manager.Queue(),
            'linear': manager.Queue(),
            'loss': manager.Queue(),
            'output': manager.Queue()
        }
        backward_queues = {
            'loss': manager.Queue(),
            'linear': manager.Queue(),
            'flatten': manager.Queue(),
            'maxpool2': manager.Queue(),
            'relu2': manager.Queue(),
            'conv2': manager.Queue(),
            'maxpool1': manager.Queue(),
            'relu1': manager.Queue(),
            'conv1': manager.Queue()
        }
    
        for i in range(num_batches):
            start = i * third_batch
            end = (i + 1) * third_batch
            currX = np.concatenate([trainX[start:end], trainX[start + third_images:end + third_images], trainX[start + 2 * third_images:end + 2 * third_images]])
            currY = np.concatenate([trainY[start:end], trainY[start + third_images:end + third_images], trainY[start + 2 * third_images:end + 2 * third_images]])

            xBatches.append(currX)
            forward_queues['conv1'].put(currX)
            yBatches.append(currY)

        workers = []
        for stage in server.get_stages():
            w1 = PipelineWorkerTwo(stage, True, server, num_batches, yBatches, learning_rate, momentum_coeff, forward_queues, backward_queues)
            w2 = PipelineWorkerTwo(stage, False, server, num_batches, yBatches, learning_rate, momentum_coeff, forward_queues, backward_queues)
            workers.append(w1)
            workers.append(w2)

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        total_idle_time = sum(server.get_idle_times())
        return server.get_metrics(trainX, trainY, pureTrainY, testX, testY, pureTestY, num_images, num_test, total_idle_time)
