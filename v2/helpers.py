import numpy as np

def im2col(X, k_height, k_width, padding=1, stride=1):
    N, C, H, W = np.shape(X)

    out_height = (H + 2*padding - k_height)//stride + 1
    out_width = (W + 2*padding - k_width)//stride + 1
    out = np.zeros((k_height, k_width, N, C, out_height, out_width), dtype=np.float32)
    
    padded_X = np.pad(X, [(0,0), (0,0), (padding, padding), (padding, padding)])

    r_end = stride*out_height
    for r in range(k_height):
        c_end = stride*out_width
        for c in range(k_width):
            out[r,c,:,:,:,:] = padded_X[:,:,r:r_end:stride,c:c_end:stride]
            c_end += 1
        r_end += 1

    out_transpose = np.transpose(out, (3,0,1,4,5,2))
    out_reshape = np.reshape(out_transpose, (C*k_height*k_width, out_height*out_width*N))
    return out_reshape

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    N, C, H, W = X_shape
    out_height = (H + 2*padding - k_height)//stride + 1
    out_width = (W + 2*padding - k_width)//stride + 1
    grad_reshape = np.reshape(grad_X_col, (C, k_height, k_width, out_height, out_width, N))
    grad_transpose = np.transpose(grad_reshape, (1, 2, 5, 0, 3, 4))

    input = np.zeros((N, C, H + 2*padding, W + 2*padding), dtype=np.float32)
    r_end = stride*out_height
    for r in range(k_height):
        c_end = stride*out_width
        for c in range(k_width):
            input[:,:,r:r_end:stride,c:c_end:stride] += grad_transpose[r,c,:,:,:,:] 
            c_end += 1
        r_end += 1

    input_no_pad = input[:, :, padding:(H + padding), padding:(W + padding)]
    return input_no_pad
