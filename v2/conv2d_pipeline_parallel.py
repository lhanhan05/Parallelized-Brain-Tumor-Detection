from modules import Transform, Conv, ReLU, MaxPool, Flatten, LinearLayer, SoftMaxCrossEntropyLoss

class ConvNetTwoPipelineParallel(Transform):
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

        self.conv1 = Conv(input_shape, filter_shape, False)
        self.relu1 = ReLU(False)
        self.maxpool1 = MaxPool((2,2),2, False)
        self.conv2 = Conv((conv_k_c,pool1_w_out,pool1_h_out), (1,5,5), False)
        self.relu2 = ReLU(False)
        self.maxpool2 = MaxPool((2,2),2, False)
        self.flatten = Flatten(False)
        self.linear = LinearLayer(conv_k_c*pool2_w_out*pool2_h_out,out_dim, False)
        self.loss = SoftMaxCrossEntropyLoss(False)

    def forward(self, inputs, y_labels):
        conv1_out = self.conv1.forward(inputs)
        relu1_out = self.relu1.forward(conv1_out)
        maxpool1_out = self.maxpool1.forward(relu1_out)
        conv2_out = self.conv2.forward(maxpool1_out)
        relu2_out = self.relu2.forward(conv2_out)
        maxpool2_out = self.maxpool2.forward(relu2_out)
        flatten_out = self.flatten.forward(maxpool2_out)
        linear_out = self.linear.forward(flatten_out)
        (losses, preds) = self.loss.forward(linear_out, y_labels, get_predictions=True)
        return (losses, preds)

    def backward(self):
        pass

    def update(self, learning_rate, momentum_coeff):
        pass

    def override_weights(self, new_weights):
        new_conv1, new_conv2, new_linear = new_weights
        conv1_weights, conv1_biases = new_conv1
        conv2_weights, conv2_biases = new_conv2
        linear_weights, linear_biases = new_linear
        self.linear.override_weights(linear_weights, linear_biases)
        self.conv1.override_weights(conv1_weights, conv1_biases)
        self.conv2.override_weights(conv2_weights, conv2_biases)

    def get_weights(self):
        return (self.conv1.get_wb_conv(), self.conv2.get_wb_conv(),self.linear.get_wb_fc())
