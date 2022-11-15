import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K

def generate_ts_data(df, label_df, input_seq_len=144, tau=48, shuffle=True):
    
    conti_input_list = []
    cate_input_list = []
    future_input_list = []
    label_list = []
    
    for i in df['year'].unique():
        tmp_df = np.array(df.loc[df['year'] == i, :])
        tmp_label_df = np.array(label_df.loc[label_df['year'] == i, ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630', 'tide_level']])
        n = tmp_df.shape[0] - input_seq_len - tau 
        
        tmp_conti_input = tmp_df[:, 4:] # (4416, 16)
        tmp_cate_input = tmp_df[:, 1:4] # (4416, 3)
        
        conti_input = np.zeros((n, input_seq_len, tmp_conti_input.shape[1]), dtype=np.float32)
        cate_input = np.zeros((n, input_seq_len, tmp_cate_input.shape[1]), dtype=np.float32)
        future_input = np.zeros((n, tau, tmp_cate_input.shape[1]), dtype=np.float32)
        label = np.zeros((n, tau, 5))
    
        for j in range(n):
            conti_input[j, :, :] = tmp_conti_input[j:(j+input_seq_len), :]
            cate_input[j, :, :] = tmp_cate_input[j:(j+input_seq_len), :]
            future_input[j, :, :] = tmp_cate_input[(j+input_seq_len):(j+input_seq_len+tau), :]
            label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

        conti_input_list.append(conti_input)
        cate_input_list.append(cate_input)
        future_input_list.append(future_input)
        label_list.append(label)
    
    total_conti_input = np.concatenate(conti_input_list, axis=0)
    total_cate_input = np.concatenate(cate_input_list, axis=0)
    total_future_input = np.concatenate(future_input_list, axis=0)
    total_label = np.concatenate(label_list, axis=0)
    
    if shuffle:
        idx = np.arange(n)
        np.random.shuffle(idx)
    else: 
        idx = np.arange(n)
    return total_conti_input[idx, ...], total_cate_input[idx, ...], total_future_input[idx, ...], total_label[idx, ...]

def get_spatial_mask(spatial_structure: list) -> tf.float32:
    cumsum_ = np.cumsum(sum(spatial_structure, []))
    spatial_mask = np.ones((cumsum_[-1], cumsum_[-1]), dtype=np.float32)
    idx = 0
    
    for element in spatial_structure:
        if len(element) == 1:  
            spatial_mask[:idx + element[0], idx + element[0]:] = 0
            idx += element[0]
        else:
            tmp_idx = idx + np.cumsum(element)
            for i, j in enumerate(tmp_idx):
                if i == 0:
                    spatial_mask[:tmp_idx[0], tmp_idx[0]:] = 0
                else:
                    spatial_mask[tmp_idx[i-1]:tmp_idx[i], idx:tmp_idx[i-1]] = 0
                    spatial_mask[:tmp_idx[i], tmp_idx[i]:] = 0
            idx += np.sum(element)        
            
    return tf.constant(spatial_mask, tf.float32)

def scaled_dot_product_attention(q, k, v, d_model, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(d_model, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + ((1 - mask) * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class QuantileRisk(K.losses.Loss):
    def __init__(self, tau, quantile, num_target):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile 
        self.q_arr = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(quantile, axis=0)[..., tf.newaxis], tau, axis=-1), 1), num_target, axis=1) # (batch_size, num_target, quantile_len, tau)

    def call(self, true, pred):
        true = tf.cast(true, tf.float32)
        true_rep = tf.repeat(tf.expand_dims(true, -1), len(self.quantile), axis=-1) # (batch_size, tau, num_target, quantile_len)
        true_rep = tf.transpose(true_rep, perm=[0, 2, 3, 1]) # (batch_size, num_target, quantile_len, tau)
        
        pred = tf.transpose(pred, perm=[0, 2, 3, 1])
        
        ql = tf.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr) * (pred - true_rep) )
        
        return tf.reduce_mean(ql)