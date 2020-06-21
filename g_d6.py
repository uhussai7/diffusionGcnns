import numpy as np
import torch
from functools import reduce

def unproject(weights):
    """
    Takes a tensor of coefficients and puts it into a convolution form
    :param waeights:
    :return:
    """

    #expected input shape is [N,7]
    N=weights.shape[1]
    kernel=torch.zeros([3,3,N],requires_grad=False)
    for i in range(0,N):
        kernel[1, 2,i] = weights[0,i]
        kernel[0, 2,i] = weights[1,i]
        kernel[0, 1,i] = weights[2,i]
        kernel[1, 0,i] = weights[3,i]
        kernel[2, 0,i] = weights[4,i]
        kernel[2, 1,i] = weights[5,i]
        kernel[1, 1,i] = weights[6,i]
    return kernel

def rotate(weights, angle,N):
    """
    Rotates weights in linear configuration
    :param weights:
    :param angle:
    :param N:
    :return:
    """

    if angle is None:
        angle = 1
    if int(angle) is False:
        raise ValueError("angles need to be ints")
    else:
        angle = angle % 6

    weights_n = weights.clone()
    for i in range(0,N):
        weights_n[0:6,i] = torch.roll(weights_n[0:6,i], angle)
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
    weights_n=weights.clone()
    for i in range(0,N):
        temp_weights = weights_n[1:3,i].clone()
        weights_n[1:3,i] = torch.roll(weights_n[4:6,i], 1)
        weights_n[4:6,i] = torch.roll(temp_weights, 1)
    return rotate(weights_n,axis,N)

def expand_scalar(weights,N):
    weights=pre_expand(weights,0,N)
    weights_e=torch.zeros([7,N,12])
    for angle in range(0,6):
        #weights_e.append(rotate(weights,angle,N))
        weights_e[:,:,angle]=rotate(weights,angle,N)
    for axis in range (0,6):
        #weights_e.append(reflect(weights,axis,N))
        weights_e[:,:,axis+6] = reflect(weights, axis, N)
    return unproject(weights_e.reshape(7,N*12))

def rotate_deep(weights,phi,N):
    #wights has size [N,12,7] where the 12 is for rotations/reflections
    #first 6 indices fo the 12 are for rotations and other six are for reflections
    #we need to shift these indices i.e. theta --> theta - phi
    weights_n=weights.clone()
    idx_rot=np.arange(6)
    idx_rot = (idx_rot - phi)%6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_rot,idx_ref))
    weights_n[:,:,:]=weights[:,idx,:]
    weights_n=weights_n.reshape(7,12*N)
    weights_n= rotate(weights_n,phi,12*N)
    weights_n=weights_n.reshape(7,12,N)
    return weights_n

def reflect_deep(weights, phi, N):
    #wights has size [N, 12,7] where the 12 is for rotations/reflections
    #first 6 indices fo the 12 are for rotations and other six are for reflections
    weights_n=weights.clone()
    idx_rot = np.arange(6)
    idx_rot = (phi-idx_rot) % 6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_ref, idx_rot))
    weights_n[:,:,:] = weights[:,idx,:]
    weights_n=weights_n.reshape(7,12*N)
    weights_n= reflect(weights_n,phi,12*N)
    return weights_n.view([7,12,N])

def expand_regular(weights,N):
    weights=pre_expand(weights,1,N)
    weights_e =torch.zeros([7,12,N,12])
    for phi in range(0,6):
        # if (phi==2):
        #     print("t")
        #     print(weights[:,0,0])
        #     temp=rotate_deep(weights.clone(), phi, N)
        #     print("n")
        #     print(temp[:,2,0])
        weights_e[:,:,:,phi]=rotate_deep(weights,phi,N)
    for phi in range(0,6):
        weights_e[:,:, :,phi+6] = reflect_deep(weights, phi, N)
    return unproject(weights_e.reshape(7,12*N*12))

def expand_bias(bias):
    shape = bias.shape
    N=shape[0]
    bias_e=torch.zeros(N*12,requires_grad=False)
    for i in range(0,N):
        for j in range(0,12):
            bias_e[12*i+j]=bias[i]
    return bias_e


def gpad(output,deep):

    # if deep==0: #this padding should already be performed but keeping here if we later generalize
    #     for c in range(0,5): #padding
    #         ct=c-1
    #         input[1:H,c*(H+1)] = np.copy(input[1,(H+1)*ct+1:(H+1)*c-1])
    #         ct=(c+2)%H
    #         input[H,(H+1)*c+1:(H+1)*c+H] =np.copy(np.flip(input[1:H,(ct+1)*(H+1)-2]))
    #if deep==1:
        #print("under construction")
        #for each real channel you have to pad each of the 12 orientation channels.
        #the padding procedure is the same as scalar EXCEPT you have to copy from a
        #different orientation channel
        #might make sense to have a pad scalar function seperately so code is not
        #repeate.

    shape = list(output.shape)
    H=shape[2]-1
    input_dim=int(shape[1]/12)
    #output_n=output.numpy()
    newshape=[shape[0],input_dim,12]+shape[-2:]
    output_n=output.view(newshape)

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
                        output_n[b, i, r,row,col,] = output_n[b, i, rt, row1,col1 ]

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
                        output_n[b, i, r, row, col] = torch.flip(output_n[b, i, rt, row1, col1],[0])

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
                    output_n[b, i, r, row, col] = output_n[b, i, rt, row1, col1]


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
                    output_n[b, i, r, row, col] = output_n[b, i, rt, row1, col1]

    #output_n=torch.vie  (output_n,shape)
    #output_n=K.variable(output_n)
    return output_n.view(shape)



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


    #print(shape)
    shape = list(kernel.shape)
    Cout = shape[0]
    Cin = shape[1]
    N = Cout * Cin
    if deep==0:
        kernel_e=expand_scalar(kernel,N)
        return post_expand(kernel_e,deep,shape)

    if deep==1:
        kernel_e = expand_regular(kernel,N)
        return post_expand(kernel_e, deep, shape)


def pre_expand(kernel,deep,N):
    if deep==0:
        kernel=kernel.permute(-1,1,0)
        kernel=kernel.reshape(7,N)
        return kernel

    if deep==1:
        kernel=kernel.permute(-1,-2,-3,-4)
        kernel=kernel.reshape(7,12,N)
        return kernel

def post_expand(kernel,deep,shape):
    if deep==0:
        Cout=shape[0]
        Cin=shape[1]
        kernel=kernel.view([3,3,Cin,Cout*12])
        kernel=kernel.permute([-1,-2,0,1])
        return kernel
    if deep==1:
        Cout=shape[0]
        Cin=shape[1]
        kernel= kernel.view(3,3,12,Cin,Cout,12)
        kernel= kernel.permute(0,1,3,2,4,5)
        kernel=kernel.reshape(3,3,Cin*12,Cout*12)
        kernel=kernel.permute(-1,-2,0,1)
        return kernel