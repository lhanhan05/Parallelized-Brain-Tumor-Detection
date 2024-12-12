from modules import Transform, Conv, ReLU, Sigmoid, LeakyReLU, MaxPool, Flatten, LinearLayer, SoftMaxCrossEntropyLoss
import numpy as np

class ConvNetOneDataParallel(Transform):
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

        self.conv = Conv(input_shape, filter_shape, True)
        self.relu = ReLU(True)
        self.sigmoid = Sigmoid(True)
        self.leakyrelu = LeakyReLU(True)
        self.maxpool = MaxPool((2,2),2, True)
        self.flatten = Flatten(True)
        self.linear = LinearLayer(self.conv_k_c*self.pool_w_out*self.pool_h_out,self.out_dim, True)
        self.loss = SoftMaxCrossEntropyLoss(True)


    def forward(self, inputs, y_labels):
        conv_out = self.conv.forward(inputs)
        relu_out, relu_x_mult = self.relu.forward(conv_out)
        maxpool_out, maxpool_grad_mask, maxpool_input_shape = self.maxpool.forward(relu_out)
        flatten_out = self.flatten.forward(maxpool_out)
        linear_out, linear_input = self.linear.forward(flatten_out)
        (losses, preds, softmax, new_labels, contrast_loss) = self.loss.forward(linear_out, y_labels, get_predictions=True)
        return (losses, preds, relu_x_mult, linear_input, softmax, new_labels, contrast_loss, maxpool_input_shape, maxpool_grad_mask)

    def backward(self, inputs, relu_x_mult, linear_input, softmax, new_labels, contrast_loss, maxpool_input_shape, maxpool_grad_mask):
        loss = self.loss.backward(softmax, new_labels, contrast_loss)
        linear_out = self.linear.backward(loss, linear_input)
        flatten_out = self.flatten.backward(linear_out[2])
        maxpool_out = self.maxpool.backward(flatten_out, maxpool_input_shape, maxpool_grad_mask)
        relu_out = self.relu.backward(maxpool_out, relu_x_mult)
        conv_out = self.conv.backward(relu_out, np.shape(inputs))
        return linear_out, conv_out
    
    def override_weights(self, new_conv, new_linear):
        conv_weights, conv_biases = new_conv
        linear_weights, linear_biases = new_linear
        self.linear.override_weights(linear_weights, linear_biases)
        self.conv.override_weights(conv_weights, conv_biases)

    def get_weights(self):
        return (self.conv.get_wb_conv() ,self.linear.get_wb_fc())
       
    def update(self, learning_rate, momentum_coeff, linear_out, conv_out, N):
        self.linear.update(learning_rate, momentum_coeff, linear_out[0], linear_out[1], N)
        self.conv.update(learning_rate, momentum_coeff, conv_out[0], conv_out[1], N)