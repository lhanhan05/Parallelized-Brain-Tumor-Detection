import numpy as np
import multiprocessing

from queue import Queue

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
    def __init__(self, stage, is_forward, model, num_batches, all_y_labels, learning_rate, momentum_coeff, forward_queues, backward_queues, shared_model):
        super().__init__()
        self.is_forward = is_forward
        self.stage = stage
        self.num_batches = num_batches
        self.all_y_labels = all_y_labels
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.model = shared_model.model
        self.shared_model = shared_model
        self.forward_queues = forward_queues
        self.backward_queues = backward_queues

    def process_forwards(self):
        for i in range(self.num_batches):
            if self.stage == 'conv':
                inputs = self.forward_queues['conv'].get()
                conv_out = self.model.conv.forward(inputs)
                self.forward_queues['relu'].put(conv_out)
            elif self.stage == 'relu':
                conv_out = self.forward_queues['relu'].get()
                acti_out = self.model.relu.forward(conv_out)
                self.forward_queues['maxpool'].put(acti_out)
            elif self.stage == 'maxpool':
                acti_out = self.forward_queues['maxpool'].get()
                maxpool_out = self.model.maxpool.forward(acti_out)
                self.forward_queues['flatten'].put(maxpool_out)
            elif self.stage == 'flatten':
                maxpool_out = self.forward_queues['flatten'].get()
                flatten_out = self.model.flatten.forward(maxpool_out)
                self.forward_queues['linear'].put(flatten_out)
            elif self.stage == 'linear':
                flatten_out = self.forward_queues['linear'].get()
                linear_out = self.model.linear.forward(flatten_out)
                self.forward_queues['loss'].put(linear_out)
            elif self.stage == 'loss':
                linear_out = self.forward_queues['loss'].get()
                losses, preds = self.shared_model.model.loss.forward(linear_out, self.all_y_labels[i], get_predictions=True)
                self.forward_queues['output'].put((losses, preds))
                self.backward_queues['loss'].put(True)
            print(f"{self.stage} forward")

    def process_backwards(self):
        for _ in range(self.num_batches):
            if self.stage == 'loss':
                _ = self.backward_queues['loss'].get()
                loss_grad = self.shared_model.model.loss.backward()
                self.backward_queues['linear'].put(loss_grad)
            elif self.stage == 'linear':
                loss_grad = self.backward_queues['linear'].get()
                linear_grad = self.model.linear.backward(loss_grad)
                self.backward_queues['flatten'].put(linear_grad)
                self.model.linear.update(self.learning_rate, self.momentum_coeff)
            elif self.stage == 'flatten':
                linear_grad = self.backward_queues['flatten'].get()
                flatten_grad = self.model.flatten.backward(linear_grad)
                self.backward_queues['maxpool'].put(flatten_grad)
            elif self.stage == 'maxpool':
                flatten_grad = self.backward_queues['maxpool'].get()
                maxpool_grad = self.model.maxpool.backward(flatten_grad)
                self.backward_queues['relu'].put(maxpool_grad)
            elif self.stage == 'relu':
                maxpool_grad = self.backward_queues['relu'].get()
                relu_grad = self.server.model.relu.backward(maxpool_grad)
                self.backward_queues['conv'].put(relu_grad)
            elif self.stage == 'conv':
                relu_grad = self.backward_queues['conv'].get()
                self.model.conv.backward(relu_grad)
                self.model.conv.update(self.learning_rate, self.momentum_coeff)
            print(f"{self.stage} backward")

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
        shared_model = manager.Namespace()
        shared_model.model = model

        for i in range(num_batches):
            start = i * third_batch
            end = (i + 1) * third_batch
            currX = np.concatenate([trainX[start:end], trainX[start + third_images:end + third_images], trainX[start + 2 * third_images:end + 2 * third_images]])
            currY = np.concatenate([trainY[start:end], trainY[start + third_images:end + third_images], trainY[start + 2 * third_images:end + 2 * third_images]])

            xBatches.append(currX)
            forward_queues['conv'].put(currX)
            yBatches.append(currY)

        workers = []
        for stage in model.get_stages():
            w1 = PipelineWorkerOne(stage, True, model, num_batches, yBatches, learning_rate, momentum_coeff, forward_queues, backward_queues, shared_model)
            w2 = PipelineWorkerOne(stage, False, model, num_batches, yBatches, learning_rate, momentum_coeff, forward_queues, backward_queues, shared_model)
            workers.append(w1)
            workers.append(w2)

        # Start workers
        for worker in workers:
            worker.start()

        # Wait for workers to finish
        for worker in workers:
            worker.join()

        print("done")

        # Collect losses from the output queue
        # for _ in range(num_batches):
        #     loss, _ = forward_queues['output'].get()
        #     losses.append(loss)

        # train_loss = np.sum(losses) / num_images
        train_loss, train_pred, _ = model.forward(trainX, trainY)
        train_accu = np.count_nonzero(train_pred == pureTrainY) / num_images
        test_loss, test_pred, _ = model.forward(testX, testY)
        test_loss = test_loss / num_test
        test_accu = np.count_nonzero(test_pred == pureTestY) / num_test

        return train_loss/num_images, train_accu, test_loss, test_accu
