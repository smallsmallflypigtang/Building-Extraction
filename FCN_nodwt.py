from convdef_fcn import  *
from non_local import  *
import numpy as np


x = tf.placeholder(tf.float32,[4,256,256,3])
y = tf.placeholder(tf.int64,[4,256,256])
is_training = tf.placeholder(tf.bool)
dropout_value = tf.placeholder(tf.float32,[])

vgg16_weights = np.load('/home/admin324/PycharmProjects/data/vgg16.npy',encoding='bytes',allow_pickle= True ).item()
print(vgg16_weights.keys())


def dwtfcn(x,is_training,dropout_value):
    # conv1
    conv1_1 = single_conv(x, name='block1',weights= vgg16_weights[b'conv1_1'])
    conv1_2 = single_conv(conv1_1 ,name='block2',weights= vgg16_weights[b'conv1_2'])
    print("conv1" + str(conv1_2.shape))

    BN1 = batch_normal(conv1_2,is_training= is_training )
    # conv2
    pool1 = pool(BN1)
    conv2_1 = single_conv(pool1, name='block3',weights= vgg16_weights[b'conv2_1'])
    conv2_2 = single_conv(conv2_1 ,name= 'block4',weights= vgg16_weights[b'conv2_2'])
    print("conv2" + str(conv2_2.shape))

    BN2 = batch_normal(conv2_2,is_training= is_training )
    # conv3
    pool2 = pool(BN2)

    conv3_1 = single_conv(pool2 ,name= 'block5',weights= vgg16_weights[b'conv3_1'])
    conv3_2 = single_conv(conv3_1 ,name= 'block6',weights= vgg16_weights[b'conv3_2'])
    conv3_3 = single_conv(conv3_2 ,name= 'block7',weights= vgg16_weights[b'conv3_3'])

    print("conv3" + str(conv3_3.shape))
    BN3 = batch_normal(conv3_3,is_training= is_training )

    #conv33 = non_local3(BN3, h=64, inputdim=256)

    # up3 = single_deconv(BN3,name= 'up_sampling3',input_dim= 256,out_dim= 2,size=4)



    # conv4

    pool3 = pool(BN3)
    conv4_1 = single_conv(pool3, name='block8',weights= vgg16_weights[b'conv4_1'])
    conv4_2 = single_conv(conv4_1, name='block9',weights= vgg16_weights[b'conv4_2'])
    conv4_3 = single_conv(conv4_2, name='block10',weights= vgg16_weights[b'conv4_3'])


    print("conv4" + str(conv4_3.shape))
    BN4 = batch_normal(conv4_3,is_training= is_training )
    # conv44 = non_local3(BN4,h=32,inputdim= 512)

    # up4 = single_deconv(BN4,name= 'up_sampling4',input_dim= 512,out_dim= 2,size=8)

    # conv5

    pool4 = pool(BN4)
    conv5_1 = single_conv(pool4, name='block11',weights= vgg16_weights[b'conv5_1'])
    conv5_2 = single_conv(conv5_1, name='block12',weights= vgg16_weights[b'conv5_2'])
    conv5_3 = single_conv(conv5_2, name='block13',weights= vgg16_weights[b'conv5_3'])
    print("conv5" + str(conv5_3.shape))

    # BN5 = tf.layers.batch_normalization(conv5_3, momentum=0.8)
    BN5 = batch_normal(conv5_3,is_training= is_training )
    # up5 = single_deconv(BN5,name= 'up_sampling5',input_dim= 512,out_dim= 2,size=16)

    # add non_local attention
    conv55 = non_local3(BN5,h=16,inputdim= 512)


    # conv6
    pool5 = pool(conv55)
    conv6_1 = conv(pool5, name='block14', input_dim=512, out_dim=4096)
    dropout6_1 = tf.nn.dropout(conv6_1, keep_prob= dropout_value)
    conv6_2 = conv(dropout6_1, name='block15', input_dim=4096, out_dim=4096)
    dropout6_2 = tf.nn.dropout(conv6_2, keep_prob= dropout_value)
    print("conv6" + str(dropout6_2.shape))




    # deconv1
    deconv1_1 = single_deconv(conv6_2, name='block16', input_dim=4096, out_dim=512,size=2)
    pool4_conv = pspconv(pool4,name= 'pool4_conv',input_dim= 512,out_dim= 512)
    deconv1_2 = deconv1_1 + pool4_conv
    fpn1 = single_deconv(deconv1_2, name='fpn1', input_dim=512, out_dim=2, size=16)
    deconv1_3 = conv(deconv1_2, name='block17', input_dim=512, out_dim=512)
    deconv1 = tf.nn.dropout(deconv1_3, keep_prob= dropout_value )
    print("deconv1" + str(deconv1_3.shape))
    # deconv2
    deconv2_1 = single_deconv(deconv1, name='block18', input_dim=512, out_dim=256,size=2)
    pool3_conv = pspconv(pool3, name='pool3_conv', input_dim=256, out_dim=256)
    deconv2_2 = deconv2_1 + pool3_conv
    fpn2 = single_deconv(deconv2_2,name= 'fpn2',input_dim=256,out_dim=2,size=8)
    deconv2_3 = conv(deconv2_2, name='block19', input_dim=256, out_dim=256)
    deconv2 = tf.nn.dropout(deconv2_3, keep_prob= dropout_value )
    print("deconv2" + str(deconv2_3.shape))
    # deconv3
    deconv3_1 = single_deconv(deconv2, name='block20', input_dim=256, out_dim=128,size=2)
    pool2_conv = pspconv(pool2, name='pool2_conv', input_dim=128, out_dim=128)
    deconv3_2 = deconv3_1 + pool2_conv
    fpn3 = single_deconv(deconv3_2 ,name= 'fpn3',input_dim= 128,out_dim= 2,size= 4)
    deconv3_3 = conv(deconv3_2, name='block21', input_dim=128, out_dim=128)
    deconv3 = tf.nn.dropout(deconv3_3, keep_prob= dropout_value )
    print("deconv3" + str(deconv3.shape))
    # deconv4
    deconv4_1 = single_deconv(deconv3, name='block22', input_dim=128, out_dim=64,size=2)
    pool1_conv = pspconv(pool1, name='pool1_conv', input_dim=64, out_dim=64)
    deconv4_2 = deconv4_1 + pool1_conv
    fpn4 = single_deconv(deconv4_2 ,name= 'fpn4',input_dim= 64,out_dim= 2,size= 2)
    deconv4_3 = conv(deconv4_2, name='block23', input_dim=64, out_dim=64)
    deconv4 = tf.nn.dropout(deconv4_3, keep_prob= dropout_value )
    print("deconv4" + str(deconv4_3.shape))
    # deconv5
    deconv5 = single_deconv(deconv4, name='block24', input_dim=64, out_dim=2,size=2)
    print("deconv5" + str(deconv5.shape))

    #fusion
    concat = tf.concat((deconv5,fpn2,fpn3,fpn4), axis=-1)
    print("concat" + str(concat.shape))
    #WU MUL
    #concat = tf.concat((deconv5), axis=-1)
    final1 = pspconv(concat, name='final1', input_dim=8, out_dim=2)


    annotation_pred = tf.argmax(final1, dimension=3,name='prediction')
    print("annotation_pred" + str(annotation_pred.shape))

    #return annotation_pred,final1
    return annotation_pred ,final1


