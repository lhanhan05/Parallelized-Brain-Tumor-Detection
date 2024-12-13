import numpy as np
from helpers import im2col, im2col_bw

class Transform:
    def __init__(self, is_data_parallel):
        self.is_data_parallel = is_data_parallel
        pass

    def forward(self, x):
        pass

    def backward(self, grad_wrt_out):
        pass

    def update(self, learning_rate, momentum_coeff):
        pass


class ReLU(Transform):
    def __init__(self, is_data_parallel):
        Transform.__init__(self, is_data_parallel)
        self.x_mult = None

    def forward(self, x):
        self.x_mult = (x > 0)
        if self.is_data_parallel:
            return x*self.x_mult, self.x_mult
        else:
            return x*self.x_mult

    def backward(self, grad_wrt_out, x_mult=None):
        if self.is_data_parallel:
            self.x_mult = x_mult
        return self.x_mult*grad_wrt_out
    
class Sigmoid(Transform):
    def __init__(self, is_data_parallel):
        Transform.__init__(self, is_data_parallel)
        self.forward_out = None

    def forward(self, x):
        self.forward_out = 1/(1 + np.exp(-1*x))
        return self.forward_out

    def backward(self, grad_wrt_out, forward_out):
        if self.is_data_parallel:
            self.forward_out = forward_out
        mult = self.forward_out*(1 - self.forward_out)
        return mult*grad_wrt_out
    
class LeakyReLU(Transform):
    def __init__(self, is_data_parallel):
        Transform.__init__(self, is_data_parallel)
        self.pos_mult = None
        self.neg_mult = None

    def forward(self, x):
        self.pos_mult = (x > 0)
        self.neg_mult = (x <= 0)*0.05
        if self.is_data_parallel:
            return x*self.pos_mult + x*self.neg_mult, self.pos_mult, self.neg_mult
        else:
            return x*self.pos_mult + x*self.neg_mult

    def backward(self, grad_wrt_out, pos_mult, neg_mult):
        if self.is_data_parallel:
            self.pos_mult = pos_mult
            self.neg_mult = neg_mult
        return self.pos_mult*grad_wrt_out + self.neg_mult*grad_wrt_out

class Dropout(Transform):
    def __init__(self, is_data_parallel, p=0.1):
        Transform.__init__(self, is_data_parallel)
        self.p = p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.binomial(1, 1 - self.p, x.shape)
        else:
            self.mask = (1 - self.p)
        return x*self.mask
    
    def backward(self, grad_wrt_out):
        return self.mask*grad_wrt_out
    
class Flatten(Transform):
    def __init__(self, is_data_parallel):
        Transform.__init__(self, is_data_parallel)

    def forward(self, x):
        self.shape = np.shape(x)
        (N,C,H,W) = self.shape
        return np.reshape(x, (N, C*H*W))

    def backward(self, dloss):
        return np.reshape(dloss, self.shape)


class Conv(Transform):
    def __init__(self, input_shape, filter_shape, is_data_parallel):
        Transform.__init__(self, is_data_parallel)
        (num_filters, k_height, k_width) = filter_shape
        (C, H, W) = input_shape
        self.num_filters = num_filters
        self.C = C
        self.H = H
        self.W = W
        self.k_height = k_height
        self.k_width = k_width

        b = np.sqrt(6/((num_filters + C)*k_height*k_width))
        self.weights = np.random.uniform(-b, b, (num_filters, C, k_height, k_width))
        self.weights_momentum = np.zeros_like(self.weights)
        self.biases = np.zeros((num_filters, 1))
        self.biases_momentum = np.zeros_like(self.biases)

    def forward(self, inputs, stride=1, pad=2):
        (N, C, H, W) = np.shape(inputs)
        self.pad = pad
        self.stride = stride
        self.N = N
        self.X = inputs
        input_cols = im2col(inputs, self.k_height, self.k_width, pad, stride)
        weights_reshape = np.reshape(self.weights, (self.num_filters, self.C*self.k_height*self.k_width))
        self.out_width = (W + 2*pad - self.k_width)//stride + 1
        self.out_height = (H + 2*pad - self.k_height)//stride + 1
        output = np.matmul(weights_reshape, input_cols) + self.biases
        output_reshape = np.reshape(output, (self.num_filters, self.out_height, self.out_width, N))
        output_transpose = np.transpose(output_reshape, (3, 0, 1, 2))
        return output_transpose

    def backward(self, dloss, input_shape=None):
        print("hi")
        print(np.shape(dloss))
        if not self.is_data_parallel:
            N, C, H, W = self.N, self.C, self.H, self.W
        else:
            N, C, H, W = input_shape
        print(N, C, H, W)
        weights_transpose = np.transpose(self.weights, (1,2,3,0))
        weights_reshape = np.reshape(weights_transpose, (self.C*self.k_height*self.k_width, self.num_filters))
        dloss_transpose = np.transpose(dloss, (1,2,3,0))
        dloss_reshape = np.reshape(dloss_transpose, (self.num_filters, self.out_height*self.out_width*self.N))
        mul_output = np.matmul(weights_reshape, dloss_reshape) 
        self.grad_inputs = im2col_bw(mul_output, (N, C, H, W), self.k_height, self.k_width, self.pad, self.stride)

        x_col = im2col(self.X, self.k_height, self.k_width, self.pad, self.stride)
        x_col_transpose = np.transpose(x_col)
        grad_weight_pre = np.matmul(dloss_reshape, x_col_transpose)
        self.grad_weights = np.reshape(grad_weight_pre, (self.num_filters, self.C, self.k_height, self.k_width))

        self.grad_biases = np.reshape(np.sum(dloss_reshape, axis=1), (self.num_filters,1))

        return [self.grad_weights, self.grad_biases, self.grad_inputs]

    def update(self, learning_rate=0.001, momentum_coeff=0.5, grad_weights=None, grad_biases=None, N=None):
        if not self.is_data_parallel:
            grad_weights = self.grad_weights
            grad_biases = self.grad_biases
            N = self.N
        self.weights_momentum = momentum_coeff*self.weights_momentum + grad_weights/N
        self.biases_momentum = momentum_coeff*self.biases_momentum + grad_biases/N
        self.weights = self.weights - learning_rate*self.weights_momentum
        self.biases = self.biases - learning_rate*self.biases_momentum  

    def override_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases
      
    def get_wb_conv(self):
        return (self.weights, self.biases)


class MaxPool(Transform):
    def __init__(self, filter_shape, stride, is_data_parallel):
        Transform.__init__(self, is_data_parallel)
        (self.k_height, self.k_width) = filter_shape
        self.stride = stride

    def forward(self, inputs):
        (self.N, self.C, self.H, self.W) = np.shape(inputs)
        self.inputs = inputs
        cols = im2col(inputs, self.k_height, self.k_width, padding=0, stride=self.stride)
        self.out_height = (self.H - self.k_height)//self.stride + 1
        self.out_width = (self.W - self.k_width)//self.stride + 1
        cols_transpose = np.reshape(cols, (self.C, self.k_height, self.k_width, self.out_height, self.out_width, self.N)).transpose(1,2,0,3,4,5)
        cols_reshape = np.reshape(cols_transpose, (self.k_height*self.k_width, self.C, self.out_height*self.out_width*self.N))
        max_forward = np.max(cols_reshape, axis=0)

        self.grad_mask = (cols_reshape == max_forward[None])
        output = np.transpose(np.reshape(max_forward, (self.C, self.out_height, self.out_width, self.N)), (3,0,1,2))
        if self.is_data_parallel:
            return output, self.grad_mask, np.shape(inputs)
        else:
            return output

    def backward(self, dloss, input_shape=None, grad_mask=None):
        if not self.is_data_parallel:
            grad_mask = self.grad_mask
            N, C, H, W = self.N, self.C, self.H, self.W
        else:
            N, C, H, W = input_shape
        cols = np.reshape(np.transpose(dloss, (1,2,3,0)), (C, self.out_height*self.out_width*self.N))
        cols_repeat = np.repeat(cols, self.k_height*self.k_width, axis=0)
        mask_reshape = np.reshape(np.transpose(grad_mask, (1,0,2)), np.shape(cols_repeat))
        grad_input = im2col_bw(cols_repeat*mask_reshape, (N, C, H, W), self.k_height, self.k_width, padding=0, stride=self.stride)
        return grad_input


class LinearLayer(Transform):
    def __init__(self, indim, outdim, is_data_parallel):
        Transform.__init__(self, is_data_parallel)
        self.indim = indim
        self.outdim = outdim
        b = np.sqrt(6)/np.sqrt(indim+outdim)
        self.weights = np.random.uniform(-b, b, (indim, outdim))
        self.weights_momentum = np.zeros_like(self.weights)

        self.biases = np.zeros((outdim,1))
        self.biases_momentum = np.zeros_like(self.biases)

    def forward(self, inputs):
        (self.N, _) = np.shape(inputs)
        self.inputs = inputs
        input_weight_prod = np.transpose(np.matmul(self.inputs, self.weights))
        if self.is_data_parallel:
            return np.transpose(input_weight_prod + self.biases), inputs
        else:
            return np.transpose(input_weight_prod + self.biases)

    def backward(self, dloss, inputs=None):
        if not self.is_data_parallel:
            inputs = self.inputs
        self.grad_weights = np.matmul(np.transpose(inputs), dloss)
        self.grad_inputs = np.matmul(dloss, np.transpose(self.weights))
        dloss_transpose = np.transpose(dloss)
        self.grad_biases = np.reshape(np.sum(dloss_transpose, axis=1), (self.outdim, 1))
        return [self.grad_weights, self.grad_biases, self.grad_inputs]

    def update(self, learning_rate, momentum_coeff, grad_weights=None, grad_biases=None, N=None):
        if not self.is_data_parallel:
            grad_weights = self.grad_weights
            grad_biases = self.grad_biases
            N = self.N
        self.weights_momentum = momentum_coeff*self.weights_momentum + grad_weights/N
        self.biases_momentum = momentum_coeff*self.biases_momentum + grad_biases/N
        self.weights = self.weights - learning_rate*self.weights_momentum
        self.biases = self.biases - learning_rate*self.biases_momentum    

    def override_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def get_wb_fc(self):
        return (self.weights, self.biases)


class SoftMaxCrossEntropyLoss(Transform):
    def __init__(self, is_data_parallel):
        Transform.__init__(self, is_data_parallel)

    def forward(self, logits, labels, get_predictions=False):
        logits = np.transpose(logits)
        self.labels = np.transpose(labels)
        logit_exp = np.exp(logits)
        expsum = np.sum(logit_exp, axis=0)
        self.softmax = logit_exp/expsum
        logsoft = np.log(self.softmax)
        loss_prod = logsoft*self.labels
        loss_sum = -1*np.sum(loss_prod)
        preds = np.transpose(np.argmax(self.softmax, axis=0))
        pred_parts = np.array_split(preds, 3)

        length = len(pred_parts[0])
        sims = [0] * length

        for i in range(length):
            element_count = {}

            for array in pred_parts:
                if i >= len(array):
                    break
                element = array[i]
                if element in element_count:
                    element_count[element] += 1
                else:
                    element_count[element] = 1

            sims[i] = max(element_count.values())
        contrast_loss_section = [-0.01*x for x in sims]
        contrast_loss = np.tile(contrast_loss_section, 3)
        contrast_sum = np.sum(contrast_loss)

        self.contrast = contrast_loss[:labels.shape[0]].reshape((-1,1)) * labels
        total_loss = 0.8*loss_sum + 0.2*contrast_sum
        if get_predictions:
            if self.is_data_parallel:
                return (total_loss, preds, self.softmax, self.labels, self.contrast)
            else:
                return (total_loss, preds)
        else:
            if self.is_data_parallel:
                return (total_loss, self.softmax, self.labels, self.contrast)
            else:
                return total_loss

    def backward(self, softmax=None, labels=None, contrast = None):
        if not self.is_data_parallel:
            softmax = self.softmax
            labels = self.labels
            contrast = self.contrast
        softmax_back = np.transpose(softmax - labels)
        return 0.8*softmax_back + 0.2*contrast

    def getAccu(self, softmax=None, labels=None):
        if self.is_data_parallel:
            self.softmax = softmax
            self.labels = labels
        preds = np.argmax(self.softmax, axis=0)
        actuals = np.argmax(self.labels, axis=0)
        correctcount = np.count_nonzero(preds==actuals)
        N = np.shape(preds)
        return correctcount/N