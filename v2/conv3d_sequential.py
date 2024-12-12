from modules import Transform, Conv, ReLU, MaxPool, Flatten, LinearLayer, SoftMaxCrossEntropyLoss, Dropout

class ConvNetThreeSequential(Transform):
    def __init__(self, out_dim=10, input_shape=(3,32,32), filter_shape=(16,3,3), dropout_prob=0.1):
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

        conv3_w_out = (pool2_w_out + 2*conv_padding - conv_k_w)//conv_stride + 1
        conv3_h_out = (pool2_h_out + 2*conv_padding - conv_k_h)//conv_stride + 1

        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.conv1 = Conv(input_shape, filter_shape)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool((2,2),2)
        self.dropout1 = Dropout(dropout_prob)

        self.conv2 = Conv((conv_k_c,pool1_w_out,pool1_h_out), filter_shape)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool((2,2),2)
        self.dropout2 = Dropout(dropout_prob)

        self.conv3 = Conv((conv_k_c,pool2_w_out,pool2_h_out), filter_shape)
        self.relu3 = ReLU()
        self.dropout3 = Dropout(dropout_prob)

        self.flatten = Flatten()
        self.linear = LinearLayer(conv_k_c*conv3_w_out*conv3_h_out,out_dim)
        self.loss = SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels):
        conv1_out = self.conv1.forward(inputs, self.conv_stride, self.conv_padding)
        relu1_out = self.relu1.forward(conv1_out)
        maxpool1_out = self.maxpool1.forward(relu1_out)
        dropout1_out = self.dropout1.forward(maxpool1_out)

        conv2_out = self.conv2.forward(dropout1_out, self.conv_stride, self.conv_padding)
        relu2_out = self.relu2.forward(conv2_out)
        maxpool2_out = self.maxpool2.forward(relu2_out)
        dropout2_out = self.dropout2.forward(maxpool2_out)

        conv3_out = self.conv3.forward(dropout2_out, self.conv_stride, self.conv_padding)
        relu3_out = self.relu3.forward(conv3_out)
        dropout3_out = self.dropout3.forward(relu3_out)

        flatten_out = self.flatten.forward(dropout3_out)
        linear_out = self.linear.forward(flatten_out)
        (losses, preds) = self.loss.forward(linear_out, y_labels, get_predictions=True)
        return (losses, preds)


    def backward(self):
        loss = self.loss.backward()
        linear_out = self.linear.backward(loss)[2]
        flatten_out = self.flatten.backward(linear_out)

        dropout3_out = self.dropout3.backward(flatten_out)
        relu3_out = self.relu3.backward(dropout3_out)
        conv3_out = self.conv3.backward(relu3_out)[2]

        dropout2_out = self.dropout2.backward(conv3_out)
        maxpool2_out = self.maxpool2.backward(dropout2_out)
        relu2_out = self.relu2.backward(maxpool2_out)
        conv2_out = self.conv2.backward(relu2_out)[2]
        
        dropout1_out = self.dropout1.backward(conv2_out)
        maxpool1_out = self.maxpool1.backward(dropout1_out)
        relu1_out = self.relu1.backward(maxpool1_out)
        conv1_out = self.conv1.backward(relu1_out)[2]
       

    def update(self, learning_rate, momentum_coeff):
        self.linear.update(learning_rate, momentum_coeff)
        self.conv3.update(learning_rate, momentum_coeff)
        self.conv2.update(learning_rate, momentum_coeff)
        self.conv1.update(learning_rate, momentum_coeff)
