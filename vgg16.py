from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops.gen_nn_ops import *

def my_profile(C,prof_type):
    def half_exp(n,k=1,dtype='float32'):
        n_ones = int(n/2)
        n_other = n - n_ones
        return np.append(np.ones(n_ones,dtype=dtype),np.exp((1-k)*np.arange(n_other),dtype=dtype))
    if prof_type == "linear":
        profile = np.linspace(1.0,0.0,num=C,endpoint=False,dtype='float32')
    elif prof_type == "all-one":
        profile = np.ones(C,dtype='float32')
    elif prof_type == "half-exp":
        profile = half_exp(C,2.0)
    elif prof_type == "harmonic":
        profile = np.array(1.0/(np.arange(C)+1))
    else:
        raise ValueError("prof_type must be \"all-one\", \"half-exp\", \"harmonic\" or \"linear\".")
    return profile

def idp_conv2d(input,filter,strides,padding,
               use_cudnn_on_gpu=True,data_format='NHWC',
               name=None,prof_type=None, dp=1.0):
    with ops.name_scope(name, "idp_convolution", [input, filter]) as scope:
        if not (data_format == "NHWC" or data_format == "NCHW"):
            raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
        
        # do conv2d
        conv2d_res = gen_nn_ops.conv2d(input,
                      filter,
                      strides,
                      padding,
                      use_cudnn_on_gpu=True,
                      data_format='NHWC',
                      name=None)
        B,H,W,C = conv2d_res.get_shape().as_list()
        
        # get profile
        profile = my_profile(C,prof_type)
        # tensor_profile = tf.get_variable(initializer=profile,name="tensor_profile",dtype='float32')

        # create a mask determined by the dot product percentage
        n1 = int(C * dp)
        n0 = C - n1
        mask = np.append(np.ones(n1,dtype='float32'),np.zeros(n0,dtype='float32'))
        if len(profile) == len(mask):
            profile *= mask
        else:
            raise ValueError("profile and mask must have the same shape.")

        # create a profile coefficient, gamma
        conv2d_profile = np.stack([profile for i in range(B*H*W)])
        conv2d_profile = np.reshape(conv2d_profile, newshape=(B,H,W,C))
        
        gamma = tf.get_variable(initializer=conv2d_profile,name="gamma"+str(dp*100))
        
        # IDP conv2d output
        idp_conv2d_res = tf.multiply(conv2d_res,gamma,name="idp"+str(dp*100))
        
        return idp_conv2d_res