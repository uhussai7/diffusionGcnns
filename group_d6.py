import numpy as np
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow import gather

# def project(kernel):
#     """
#     Projects onto kernel basis. Fancy way of saying extract coefficients
#     :param kernel: this will have shape 3 x 3 x (N_shells x N_filters) but has to be numpy
#     :return: weights this will have shape 6 x (N_shells x N_filters) will be numpy
#     """
#     N=kernel.shape[-1]
#     weights = np.empty([7,N])
#     for i in range(0,N):
#         weights[0,i] = kernel[1, 2, i]
#         weights[1,i] = kernel[0, 2, i]
#         weights[2,i] = kernel[0, 1, i]
#         weights[3,i] = kernel[1, 0, i]
#         weights[4,i] = kernel[2, 0, i]
#         weights[5,i] = kernel[2, 1, i]
#         weights[6, i] = kernel[1, 1, i]
#
#     return weights


def unproject(weights):
    """
    Takes coefficients and puts it back in kernel form
    :param weights: has shape 6 x (N_shells x N_filters) but has to be numpy
    :return: kernel 3 x 3 x (N_shells x N_filters) but has to be numpy
    """
    N=weights.shape[-1]
    kernel = np.zeros([3,3,N])
    for i in range(0,N):
        kernel[1, 2,i] = weights[0,i]
        kernel[0, 2,i] = weights[1,i]
        kernel[0, 1,i] = weights[2,i]
        kernel[1, 0,i] = weights[3,i]
        kernel[2, 0,i] = weights[4,i]
        kernel[2, 1,i] = weights[5,i]
        kernel[1, 1,i] = weights[6, i]
    return kernel

def rotate(weights, angle,N):
    """
    Counter clockwise rotation
    :param weights: numpy array of 7 weights, center weight at end
    :param angle: angle is represented as an integer
    :param N: total number of filters
    :return: rotated weights
    """
    if angle is None:
        angle = 1
    if int(angle) is False:
        raise ValueError("angles need to be ints")
    else:
        angle = angle % 6

    #weights_n = np.empty([7, N])
    weights_n = np.copy(weights)
    for i in range(0,N):
        weights_n[0:6,i] = np.copy(np.roll(weights[0:6,i], angle))
    return weights_n


def reflect(weights, axis,N):
    """
    Reflects along an axis.
    :param weights: numpy array of 7 weights, center weight at end
    :param axis: axis of reflection, represented as integer 0 is x-axis, 1 is middle of first edge, 2 is first vertex etc..
    :param N: total number of filters
    :return: rotated weights
    """
    if axis is None:
        axis = 0
    if int(axis) is False:
        raise ValueError("axis need to be ints")
    else:
        axis = axis % 6

    # first reflect on x-axis and then rotate
    #weights_n=np.zeros([7,N])
    weights_n=np.copy(weights)
    for i in range(0,N):
        temp_weights = np.copy(weights[1:3,i])
        weights_n[1:3,i] = np.copy(np.roll(weights[4:6,i], 1))
        weights_n[4:6,i] = np.copy(np.roll(temp_weights, 1))
    return rotate(weights_n,axis,N)


def expand_scalar(weights,N):

    weights_e = np.zeros([7,N,12])
    #weights_e = []
    for angle in range(0,6):
        #weights_e.append(rotate(weights,angle,N))
        weights_e[:,:,angle]=rotate(weights,angle,N)
    for axis in range (0,6):
        #weights_e.append(reflect(weights,axis,N))
        weights_e[:, :, axis+6] = reflect(weights, axis, N)
    return weights_e

def expand_regular(weights,N):
    weights_e = np.zeros([7,12,N,12])
    for phi in range(0,6):
        weights_e[:,:,:,phi]= rotate_deep(weights,phi,N)
    for phi in range(0,6):
        weights_e[:, :, :,phi+6] = reflect_deep(weights, phi, N)
    return weights_e

def rotate_deep(weights,phi,N):
    #wights has size [7,12,N] where the 12 is for rotations/reflections
    #first 6 indices fo the 12 are for rotations and other six are for reflections
    #we need to shift these indices i.e. theta --> theta - phi

    idx_rot=np.arange(6)
    idx_rot = (idx_rot - phi)%6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_rot,idx_ref))
    weights=weights[:,idx,:]
    weights=np.reshape(weights,[7,12*N])
    weights= rotate(weights,phi,N)
    return np.reshape(weights,[7,12,N])

def reflect_deep(weights, phi, N):
    #wights has size [7,12,N] where the 12 is for rotations/reflections
    #first 6 indices fo the 12 are for rotations and other six are for reflections

    idx_rot = np.arange(6)
    idx_rot = (phi-idx_rot) % 6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_ref, idx_rot))
    weights = weights[:, idx, :]
    weights = np.reshape(weights, [7, 12 * N])
    weights = reflect(weights, phi,N)
    return np.reshape(weights, [7, 12, N])



def gpad(output,deep):
    shape=output.shape.as_list()
    H=shape[1]-1
    # if deep==0: #this padding should already be performed but keeping here if we later generalize
    #     for c in range(0,5): #padding
    #         ct=c-1
    #         input[1:H,c*(H+1)] = np.copy(input[1,(H+1)*ct+1:(H+1)*c-1])
    #         ct=(c+2)%H
    #         input[H,(H+1)*c+1:(H+1)*c+H] =np.copy(np.flip(input[1:H,(ct+1)*(H+1)-2]))
    if deep==1:
        #print("under construction")
        #for each real channel you have to pad each of the 12 orientation channels.
        #the padding procedure is the same as scalar EXCEPT you have to copy from a
        #different orientation channel
        #might make sense to have a pad scalar function seperately so code is not
        #repeate.

        shape=output.shape.as_list()
        input_dim=int(shape[-1]/12)
        output_n=output.numpy()
        newshape=shape[0:-1]+[input_dim,12]
        output_n=np.reshape(output_n,newshape)

        strip=np.arange(H)
        strips = np.arange(H-1)
        CW=H+1

        for b in range(0,shape[0]): #handle the batch size first off
            for i in range(0,input_dim):
                for r in range(0,12): #this is each orientataion channel
                    for c in range(0,5):

                        if c==0: #left
                            ct = c - 1
                            rot_dir=1 #change to minus if you want rotation the other way
                            if r<=5: #left
                                rt=(r+rot_dir)%6
                            if r>5:
                                rt=((r-6+rot_dir) % 6) + 6
                            row=1+strips
                            col=c*CW
                            row1 = 1
                            col1 = ct*CW+1+strips
                            #output_n[b,1:H,c*(H+1),i,r] = np.copy(output_n[b,1,(H+1)*ct+1:(H+1)*c-1,i,rt])
                            output_n[b, row,col, i, r] = np.copy(output_n[b, row1,col1 , i, rt])

                        if c==4: #right
                            ct=(c+3) %H
                            if r <= 5:  # next two if statements for reflection padding
                                rt = ((2 - r) % 6) + 6
                            if r <= 5:
                                rt = (2 - (r - 6)) % 6
                            row = strip
                            col = -1
                            row1 = H-1
                            col1 = ct*CW+1+ strip
                            #output_n[b,0:H,-1,i,r]= np.flip(np.copy(output_n[b,H-1,ct * (H + 1)+1:ct * (H + 1)+1+H,i,rt]))
                            output_n[b, row, col, i, r] = np.flip(np.copy(output_n[b, row1, col1, i, rt]))

                        rot_dir = -1 #top  # change to minus if you want rotation the other way
                        ct = (c + 1)%H
                        if r <= 5:
                            rt = (r + rot_dir) % 6
                        if r > 5:
                            rt = ((r - 6 + rot_dir) % 6) + 6
                        row = 0
                        col = c*CW+1+strip
                        row1 = strip
                        col1 = ct*CW+1
                        #output_n[b, 0, c * (H + 1)+1:c * (H + 1)+1+H, i, r] = np.copy(output_n[b, 0:H, ct*(H+1) , i, rt])
                        output_n[b, row, col, i, r] = np.copy(output_n[b, row1, col1, i, rt])


                        ct = (c + 3)%H #bottom
                        if r<=5: #next two if statements for reflection padding
                            rt=((2-r)%6)+6
                        if r<=5:
                            rt=(2-(r-6))%6
                        row = H
                        col = c*CW + 1+strips
                        row1 = 1+strips
                        col1 = ct * CW -2
                        #output_n[b,H, (H + 1) * c + 1:(H + 1) * c + H,i,r] = np.copy(np.flip(output_n[b,1:H, (ct + 1) * (H + 1) - 2,i,rt]))
                        output_n[b, row, col, i, r] = np.copy(output_n[b, row1, col1, i, rt])

        output_n=np.reshape(output_n,shape)
        return K.variable(output_n)

def conv2d(input,kernel,deep):
    """
    Takes group convolutions where the group is the dihedral group of order 12
    :param input: scalar or regular input, note that scalar layers need to be padded before hand
    :param kernel: the kernel
    :param deep: whether it is a deep layer or the first layer
    :return: the result of the convolution
    """
#     #have to differentiate between scalar or regular (thats tricky)
#     # if input has size [batch,x,y,shells] it is scalar

    #this is for scalar
    #expand the kernels
    #assume that kernel has shape [3,3,in,out] and out is the number of channels
    #in = 3
    #we have to dump the froup expansion into the channel dimension

    shape=kernel.shape
    Ns = kernel.shape[-2:]
    N = reduce(lambda x, y: x * y, Ns)
    if deep==0:
        kernel = K.reshape(kernel, [7, N])
        kernel_e=expand_scalar(kernel.numpy(),N)
        new_shape=shape.as_list()
        new_shape[0]=3
        new_shape[-1]=12*new_shape[-1]
        new_shape=[3,]+ new_shape
        kernel_e=np.reshape(kernel_e,[7,12*N])
        kernel_e=unproject(kernel_e)
        kernel_e=np.reshape(kernel_e,new_shape)
        kernel_e=K.variable(kernel_e)
        return gpad(K.conv2d(input, kernel_e,padding="same"), 1)
        #return kernel_e

    if deep==1:
        kernel = K.reshape(kernel, [7,12,N])
        kernel_e=expand_regular(kernel.numpy(),N)
        new_shape = shape.as_list()
        new_shape[0]=3
        new_shape[-1] = 12 * new_shape[-1]
        new_shape=[3,] +new_shape
        kernel_e=np.reshape(kernel_e,[7,12*12*N])
        kernel_e=unproject(kernel_e)
        kernel_e=np.reshape(kernel_e,new_shape)
        kernel_e=K.variable(kernel_e)
        #return kernel_e
        return gpad(K.conv2d(input,kernel_e,padding="same"),deep)

#     # if input has size [batch, x,y,shells*12] it is regular
#     #kernel will have size [x,y,shells,filters]
