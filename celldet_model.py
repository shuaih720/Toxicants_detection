import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers,Sequential,Model
from tabnet.custom_objects import glu, GroupNormalization
import tensorflow_addons as tfa
import sklearn
import sklearn.metrics as metrics
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import os
from tensorflow.keras.optimizers import SGD,Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.io import loadmat
from scipy import interpy
from itertools import cycle
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
# import h5py

class TemporalAttention(layers.Layer):
    def __init__(self, feature_dim,kernel_size,stride,name='',**kwargs):
        super(TemporalAttention,self).__init__(name=name,**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=feature_dim,kernel_size=kernel_size,strides=stride, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
    def call(self,x):
        att = self.conv(x)
        att = self.bn(att)
        att = self.sigmoid(att)
        
        y = att * x
        
        return y
    
#1.posi_encoding
#输入为每个data module在数据中的位置
def channel_encoding(channel,d_model):
    """
    :param channel: channel position in original ECG data
    :param d_model: inner_dim,related to num_units
    :return: channel_encoding
    """
    def get_angles(channel,i):
        return channel / np.power(10000., 2. * (i // 2.) / np.float(d_model))
    angle_rates = get_angles(np.arange(channel)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    # operated by sin() in 2*i position, and by cos() in 2*i+1 position
    channel_sin = np.sin(angle_rates[:,0::2])
    channel_cos = np.cos(angle_rates[:,1::2])
    channel_encoding = np.concatenate([channel_sin,channel_cos],axis=-1)
    channel_encoding = tf.cast(channel_encoding[np.newaxis,...],tf.float32)
    return channel_encoding

#2.Multi-Head Attention
def scaled_dot_product_attention(q, k, v):
    '''attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V'''
    matmul_QK = tf.matmul(q,k,transpose_b=True)
    dk = tf.cast(tf.shape(q)[-1],tf.float32)
    scaled_attention = matmul_QK / tf.math.sqrt(dk)
    # 掩码mask
    # if mask is not None:
    #     # 这里将mask的token乘以-1e-9，这样与attention相加后，mask的位置经过softmax后就为0
    #     # padding位置 mask=1
        # scaled_attention += mask * -1e-9
    # 通过softmax获取attention权重, mask部分softmax后为0
    attention_weights = tf.nn.softmax(scaled_attention)  # shape=[batch_size, seq_len_q, seq_len_k]
    # 乘以value
    outputs = tf.matmul(attention_weights, v)  # shape=[batch_size, seq_len_q, depth]
    return outputs, attention_weights
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        # d_model必须可以正确分成多个头
        assert d_model % num_heads == 0
        # 分头之后维度
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        # 分头，将头个数的维度，放到seq_len前面 x输入shape=[batch_size, seq_len, d_model]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        # 分头前的前向网络，根据q,k,v的输入，计算Q, K, V语义
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        # 分头
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)
        
        # 通过缩放点积注意力层
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]
        # 把多头合并
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model]
        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights

#3.Layer Normalization implementation
class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
    
#4. feedforward
def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation=tf.nn.relu), 
        tf.keras.layers.Dense(d_model)
    ])

#5
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1,name=''):
        super(EncoderLayer, self).__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training,mask=32):
        # multi head attention (encoder时Q = K = V)
        print(inputs.shape)
        att_output, _ = self.mha(inputs, inputs, inputs,mask)
        att_output = self.dropout1(att_output, training=training)
        print(att_output.shape)
        output1 = self.layernorm1(inputs + att_output)  # shape=[batch_size, seq_len, d_model]
        # feed forward network
        ffn_output = self.ffn(output1)
        # print('3_',ffn_output.shape)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # shape=[batch_size, seq_len, d_model]
        return output2

#Tabnet implementation
class TabNet(tf.keras.layers.Layer):
    
    def __init__(self, num_features,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=2,
                 epsilon=1e-5,
                 **kwargs):
        
        super(TabNet, self).__init__(**kwargs)

        # Input checks
        # if feature_columns is not None:
        #     if type(feature_columns) not in (list, tuple):
        #         raise ValueError("`feature_columns` must be a list or a tuple.")

        #     if len(feature_columns) == 0:
        #         raise ValueError("`feature_columns` must be contain at least 1 tf.feature_column !")

        #     if num_features is None:
        #         num_features = len(feature_columns)
        #     else:
        #         num_features = int(num_features)

        # else:
        #     if num_features is None:
        #         raise ValueError("If `feature_columns` is None, then `num_features` cannot be None.")

        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        if feature_dim <= output_dim:
            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        num_groups = max(1, int(num_groups))
        epsilon = float(epsilon)

        if relaxation_factor < 0.:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        if norm_type not in ['batch', 'group']:
            raise ValueError("`norm_type` must be either `batch` or `group`")

        self.feature_columns = feature_columns=None
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim     
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.norm_type = norm_type
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)

            if self.norm_type == 'batch':
                self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')
            else:
                self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')

        else:
            self.input_features = None
            self.input_bn = None

        self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f1')

        self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f2')

        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f3_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f4_{i}')
            for i in range(self.num_decision_steps)
        ]
        
        self.transform_coef_list = [
            TransformBlock(self.num_features, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'coef_{i}')
            for i in range(self.num_decision_steps - 1)
        ]
        
        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)
            
        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.


        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below.=
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)

            if (ni > 0 or self.num_decision_steps == 1):
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim:]

            if ni < (self.num_decision_steps - 1):
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = self.transform_coef_list[ni](features_for_coef, training=training)
                # print(mask_values.shape,'-----------------------------------',type(mask_values))
                mask_values *= complementary_aggregated_mask_values
                mask_values = tfa.activations.sparsemax(mask_values, axis=-1)
                # print('##################################################')

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (
                                     tf.cast(self.num_decision_steps - 1, tf.float32))

                # Add entropy loss
                entropy_loss = total_entropy

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

                # Visualization of the feature selection mask at decision step ni
                # tf.summary.image(
                #     "Mask for step" + str(ni),
                #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                #     max_outputs=1)
                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                # This branch is needed for correct compilation by tf.autograph
                entropy_loss = 0.
        
        # Adds the loss automatically
        self.add_loss(self.sparsity_coefficient * entropy_loss)

        # Visualization of the aggregated feature importances
        # tf.summary.image(
        #     "Aggregated mask",
        #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
        #     max_outputs=1)

        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask
        
        return output_aggregated, self._step_aggregate_feature_selection_mask

    @property
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask

#TabNetClassifier implementation
class TabNetClassifier(tf.keras.layers.Layer):

    def __init__(self,num_features,
                 num_classes,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,#3-5
                 relaxation_factor=1.5,#2-3
                 sparsity_coefficient=1e-5,#delete
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 **kwargs):
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             **kwargs)
                             

        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')
        
    def call(self, inputs, training=None):
        out = self.tabnet(inputs, training=training)
        # self.activations = self.tabnet(inputs, training=training)
        # out = self.clf(self.activations)
        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)

# Aliases
TabNetClassification = TabNetClassifier

#TransformBlock implementation
class TransformBlock(tf.keras.layers.Layer):

    def __init__(self, features,
                 norm_type='batch',
                 momentum=0.9,
                 virtual_batch_size=None,
                 groups=2,
                 block_name='',
                 **kwargs):
        super(TransformBlock, self).__init__(**kwargs)

        self.features = features
        self.norm_type = norm_type
        self.momentum = momentum
        self.groups = groups
        self.virtual_batch_size = virtual_batch_size

        self.transform = tf.keras.layers.Dense(self.features, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                         virtual_batch_size=virtual_batch_size)

        # if norm_type == 'batch':
        #     self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
        #                                                  virtual_batch_size=virtual_batch_size,
        #                                                  name=f'transformblock_bn_{block_name}')

        # else:
        #     self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')

    def call(self, inputs, training=None):
        # print(inputs.shape,'---------------------------------------------------------',type(inputs))
        x = self.transform(inputs)
        
        x = self.bn(x, training=training)
        return x

from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D,Activation,AveragePooling2D,Flatten,BatchNormalization,MaxPooling2D,Dense
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
 
    # 64,64,256
    filters1, filters2, filters3 = filters
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    # 降维
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
 
    # 3x3卷积
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
 
    # 升维
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
 
    # 残差边
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
 
 
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
 
    filters1, filters2, filters3 = filters
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    # 降维
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3x3卷积
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 升维
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
 
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
       
def cellnet(classes=3):
    # [224,224,3]
    img_input = Input(shape=[680,680,3],name='input_signal')
    x = ZeroPadding2D((3, 3))(img_input)   # [230,230,3]
    # [112,112,64]
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)   #[112,112,64]
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
 
    # [56,56,64]
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
 
    # [56,56,256]
    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='c')
    
    # [28,28,512]
    x = conv_block(x, 3, [32, 32, 128], stage=3, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='c')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='d')
    
#     print(x.shape)
    # [14,14,1024]
    x = conv_block(x, 3, [64, 64, 256], stage=4, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='d')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='e')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='f')
 
#     # [7,7,2048]
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    print(x.shape)
    
#     x = EncoderLayer(d_model=128,num_heads=8, dff=128, dropout_rate=0.2,name='signal_block_trans')(x)
#     x = TemporalAttention(feature_dim=128,kernel_size=3,stride=1,name = 'signal_tem_atten')(x)
#     fla = tf.keras.layers.Flatten(name='signal_fla')(x)
    
#     cell_tab1 = TabNet(num_features=960,feature_dim=256,output_dim=128,num_decision_steps=1,relaxation_factor=3,norm_type='group',
#     batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='signal_block_tab')(fla)[0]#num_features=960
    
#     input_clinical = tf.keras.Input(shape=7,name='input_clinical')
    
#     feature_tab1 = TabNet(num_features=7,feature_dim=128,output_dim=64,num_decision_steps=2,relaxation_factor=3,norm_type='group',
#     batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='clinical_block_tab')(input_clinical)[0]
    
#     fusion_block = tf.keras.layers.Concatenate(axis = -1,name = 'fusion_concat')([cell_tab1,input_clinical])

    fla = tf.keras.layers.Flatten(name='signal_fla')(x)
    cell_tab1 = TabNet(num_features=960,feature_dim=1024,output_dim=128,num_decision_steps=1,relaxation_factor=3,norm_type='group',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='signal_block_tab')(fla)[0]#num_features=960
    
#     fusion_block_1 = tf.keras.layers.Dense(512 , activation='relu',name='fusion_dense1')(cell_tab1)
#     fusion_block_2 = tf.keras.layers.Dropout(rate = 0.2,name='fusion_dropout1')(fusion_block_1)

#     fusion_block_3 = tf.keras.layers.Dense(64 , activation='relu',name='fusion_dense2')(fusion_block_2)
#     fusion_block_4 = tf.keras.layers.Dropout(rate = 0.2,name='fusion_dropout2')(fusion_block_3)
    
    input_clinical = tf.keras.Input(shape=7,name='input_clinical')
    
    feature_tab1 = TabNet(num_features=7,feature_dim=32,output_dim=16,num_decision_steps=2,relaxation_factor=3,norm_type='group',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='clinical_block_tab')(input_clinical)[0]
    
    fusion_block = tf.keras.layers.Concatenate(axis = -1,name = 'fusion_concat')([cell_tab1,feature_tab1])
   
    fusion_block_3 = tf.keras.layers.Dense(classes, activation='softmax',name='fusion_dense3')(fusion_block)
    
    model = tf.keras.models.Model([img_input,input_clinical],fusion_block_3)
    
    return model

with tf.device("/gpu:0"):
    loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0.30,reduction=tf.keras.losses.Reduction.AUTO)
#     input_shape = (680, 680, 3)  # 输入图片尺寸
#     num_classes = 3  # 类别数
    model=cellnet()
#     model.summary()
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=300)
    adam = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit([train_data,train_feature_data],label_all_onehot, batch_size=32, epochs=20,validation_data=([x_val,x_feature_val],y_val_onehot),verbose=1,shuffle=True)