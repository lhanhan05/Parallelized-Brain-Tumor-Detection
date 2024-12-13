from modules import Transform, Conv, ReLU, Sigmoid, LeakyReLU, MaxPool, Flatten, LinearLayer, SoftMaxCrossEntropyLoss
from queue import Queue

class ConvNetOnePipelineParallel(Transform):
    def __init__(self, out_dim=20, input_shape=(3,32,32), filter_shape=(1,5,5)):
        self.out_dim = out_dim
        (_, conv_w, conv_h) = input_shape
        (conv_padding, conv_stride) = (2,1)
        (self.conv_k_c, conv_k_w, conv_k_h) = filter_shape
        pool_stride = 2
        (pool_w, pool_h) = (2,2)
        self.conv_w_out = (conv_w + 2*conv_padding - conv_k_w)//conv_stride + 1
        self.conv_h_out = (conv_h + 2*conv_padding - conv_k_h)//conv_stride + 1

        self.pool_w_out = (self.conv_w_out - pool_w)//pool_stride  + 1
        self.pool_h_out = (self.conv_h_out - pool_h)//pool_stride  + 1

        self.conv = Conv(input_shape, filter_shape, False)
        self.relu = ReLU(False)
        self.sigmoid = Sigmoid(False)
        self.leakyrelu = LeakyReLU(False)
        self.maxpool = MaxPool((2,2),2, False)
        self.flatten = Flatten(False)
        self.linear = LinearLayer(self.conv_k_c*self.pool_w_out*self.pool_h_out,self.out_dim, False)
        self.loss = SoftMaxCrossEntropyLoss(False)
        self.stages = ['loss', 'linear', 'flatten', 'maxpool', 'relu', 'conv']

    def forward(self, inputs, y_labels):
        conv_out = self.conv.forward(inputs)
        acti_out = self.relu.forward(conv_out)
        maxpool_out = self.maxpool.forward(acti_out)
        flatten_out = self.flatten.forward(maxpool_out)
        linear_out = self.linear.forward(flatten_out)
        (losses, preds) = self.loss.forward(linear_out, y_labels, get_predictions=True)
        return (losses, preds)

    def backward(self, learning_rate, momentum_coeff):
        pass

    def update(self, learning_rate, momentum_coeff):
        pass

    def get_stages(self):
        return self.stages