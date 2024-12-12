from modules import Transform, Conv, ReLU, MaxPool, Flatten, LinearLayer, SoftMaxCrossEntropyLoss
import numpy as np

class ConvNetTwoDataParallel(Transform):
    def __init__(self, out_dim=20, input_shape=(3,32,32), filter_shape=(1,5,5)):
        (conv_c, conv_w, conv_h) = input_shape
        (conv_padding, conv_stride) = (2,1)
        (conv_k_c, conv_k_w, conv_k_h) = filter_shape
        pool_stride = 2
        (pool_w, pool_h) = (2,2)
        conv1_w_out = (conv_w + 2*conv_padding - conv_k_w)//conv_stride + 1
        conv1_h_out = (conv_h + 2*conv_padding - conv_k_h)//conv_stride + 1

        pool1_w_out = (conv1_w_out - pool_w)//pool_stride  + 1
        pool1_h_out = (conv1_h_out - pool_h)//pool_stride  + 1

        conv2_w_out = (pool1_w_out + 2*conv_padding - conv_k_w)//conv_stride + 1
        conv2_h_out = (pool1_h_out + 2*conv_padding - conv_k_h)//conv_stride + 1

        pool2_w_out = (conv2_w_out - pool_w)//pool_stride  + 1
        pool2_h_out = (conv2_h_out - pool_h)//pool_stride  + 1

        self.conv1 = Conv(input_shape, filter_shape, True)
        self.relu1 = ReLU(True)
        self.maxpool1 = MaxPool((2,2),2, True)
        self.conv2 = Conv((conv_k_c,pool1_w_out,pool1_h_out), (1,5,5), True)
        self.relu2 = ReLU(True)
        self.maxpool2 = MaxPool((2,2),2, True)
        self.flatten = Flatten(True)
        self.linear = LinearLayer(conv_k_c*pool2_w_out*pool2_h_out,out_dim, True)
        self.loss = SoftMaxCrossEntropyLoss(True)

    def forward(self, inputs, y_labels):
        conv1_out = self.conv1.forward(inputs)
        relu1_out, relu1_x_mult = self.relu1.forward(conv1_out)
        maxpool1_out, maxpool1_grad_mask, maxpool1_input_shape = self.maxpool1.forward(relu1_out)
        conv2_out = self.conv2.forward(maxpool1_out)
        relu2_out, relu2_x_mult = self.relu2.forward(conv2_out)
        maxpool2_out, maxpool2_grad_mask, maxpool2_input_shape = self.maxpool2.forward(relu2_out)
        flatten_out = self.flatten.forward(maxpool2_out)
        linear_out, linear_input = self.linear.forward(flatten_out)
        (losses, preds, softmax, new_labels, contrast_loss) = self.loss.forward(linear_out, y_labels, get_predictions=True)
        parallel_updates = (np.shape(inputs), np.shape(maxpool1_out),relu1_x_mult, maxpool1_grad_mask, maxpool1_input_shape, relu2_x_mult, maxpool2_grad_mask, maxpool2_input_shape, linear_input, softmax, new_labels, contrast_loss)
        return (losses, preds, parallel_updates)

    def backward(self, parallel_updates):
        (input_shape, conv2_input_shape, relu1_x_mult, maxpool1_grad_mask, maxpool1_input_shape, relu2_x_mult, maxpool2_grad_mask, maxpool2_input_shape, linear_input, softmax, new_labels, contrast_loss) = parallel_updates
        loss = self.loss.backward(softmax, new_labels, contrast_loss)
        linear_out = self.linear.backward(loss, linear_input)
        flatten_out = self.flatten.backward(linear_out[2])

        maxpool2_out = self.maxpool2.backward(flatten_out, maxpool2_input_shape, maxpool2_grad_mask)
        relu2_out = self.relu2.backward(maxpool2_out, relu2_x_mult)
        conv2_out = self.conv2.backward(relu2_out, conv2_input_shape)

        maxpool1_out = self.maxpool1.backward(conv2_out[2], maxpool1_input_shape, maxpool1_grad_mask)
        relu1_out = self.relu1.backward(maxpool1_out, relu1_x_mult)
        conv1_out = self.conv1.backward(relu1_out, input_shape)

        return linear_out, conv1_out, conv2_out
    
    def override_weights(self, new_weights):
        new_conv1, new_conv2, new_linear = new_weights
        conv1_weights, conv1_biases = new_conv1
        conv2_weights, conv2_biases = new_conv2
        linear_weights, linear_biases = new_linear
        self.linear.override_weights(linear_weights, linear_biases)
        self.conv1.override_weights(conv1_weights, conv1_biases)
        self.conv2.override_weights(conv2_weights, conv2_biases)

    def update(self, learning_rate, momentum_coeff, outputs, N):
        (linear_out, conv1_out, conv2_out) = outputs
        self.linear.update(learning_rate, momentum_coeff, linear_out[0], linear_out[1], N)
        self.conv1.update(learning_rate, momentum_coeff, conv1_out[0], conv1_out[1], N)
        self.conv2.update(learning_rate, momentum_coeff, conv2_out[0], conv2_out[1], N)

    def get_weights(self):
        return (self.conv1.get_wb_conv(), self.conv2.get_wb_conv(),self.linear.get_wb_fc())
