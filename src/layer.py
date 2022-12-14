import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from src.utils import *

class ContiFeatureEmbedding(K.layers.Layer):
    def __init__(self, d_embedding, num_rv):
        super(ContiFeatureEmbedding, self).__init__()
        self.d_embedding = d_embedding
        self.num_rv = num_rv
        self.dense_layers = [K.layers.Dense(d_embedding) for _ in range(num_rv)]
        
    def call(self, x):
        tmp_feature_list = []                    

        for i in range(self.num_rv):
            tmp_feature = self.dense_layers[i](x[:, :, i:i+1])            
            tmp_feature_list.append(tmp_feature)

        feature_list = tf.concat(tmp_feature_list, axis=-1) ### (batch_size, time_step, num_rv * d_embedding)
        
        return tf.reshape(feature_list, [feature_list.shape[0], feature_list.shape[1], self.num_rv, self.d_embedding])

class CateFeatureEmbedding(K.layers.Layer):
    def __init__(self, d_embedding, cat_dim):
        super(CateFeatureEmbedding, self).__init__()
        self.d_embedding = d_embedding
        self.embedding_layer = K.layers.Embedding(input_dim=cat_dim, output_dim=d_embedding)
        
    def call(self, x):
        return tf.expand_dims(self.embedding_layer(x), axis=2)
    
class GLULN(K.layers.Layer):
    def __init__(self, d_model):
        super(GLULN, self).__init__()    
        self.dense1 = K.layers.Dense(d_model, activation='sigmoid')
        self.dense2 = K.layers.Dense(d_model)
        self.layer_norm = K.layers.LayerNormalization()

    def call(self, x, y):
        return self.layer_norm(tf.keras.layers.Multiply()([self.dense1(x),
                                        self.dense2(x)]) + y)

class GatedResidualNetwork(K.layers.Layer):
    def __init__(self, d_model, dr): 
        super(GatedResidualNetwork, self).__init__()        
        self.dense1 = K.layers.Dense(d_model, activation='elu')        
        self.dense2 = K.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dr)
        self.glu_and_layer_norm = GLULN(d_model)
        
    def call(self, a):
        eta_2 = self.dense1(a)
        eta_1 = self.dropout(self.dense2(eta_2))
        grn_output = self.glu_and_layer_norm(eta_1, eta_2)
        
        return grn_output

class VariableSelectionNetwork(K.layers.Layer):
    def __init__(self, d_model, d_input, dr):
        super(VariableSelectionNetwork, self).__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.dr = dr
        self.v_grn = GatedResidualNetwork(d_input, dr)
        self.softmax = K.layers.Softmax()
        self.xi_grn = [GatedResidualNetwork(d_model, dr) for _ in range(self.d_input)]
 
    def call(self, xi):
        
        Xi = tf.reshape(xi, [xi.shape[0], xi.shape[1], -1])
        weights = tf.expand_dims(self.softmax(self.v_grn(Xi)), axis=-1)
            
        tmp_xi_list = []                    
        
        for i in range(self.d_input):
            tmp_xi = self.xi_grn[i](xi[:, :, i:i+1, :])            
            tmp_xi_list.append(tmp_xi)
        
        xi_list = tf.concat(tmp_xi_list, axis=2)
        combined = tf.keras.layers.Multiply()([weights, xi_list]) # attention

        vsn_output = tf.reduce_sum(combined, axis=2) 
    
        return vsn_output
    
class SpatialStructureAttention(K.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, spatial_structure: list):
        super(SpatialStructureAttention, self).__init__()
        self.spatial_structure = get_spatial_mask(spatial_structure)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size, seq_len):
        x = tf.reshape(x, (batch_size, seq_len, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def call(self, q, k, v):
        q = self.wq(q)  # (batch_size, seq_len, num_var, d_model)
        k = self.wk(k)  # (batch_size, seq_len, num_var, d_model)
        v = self.wv(v)  # (batch_size, seq_len, num_var, d_model)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.d_model, self.spatial_structure)

        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, num_var, d_model)

        return output, attention_weights
    
class TemporalStructureAttention(K.layers.Layer):
    def __init__(self, d_model, num_heads, seq_len):
        super(TemporalStructureAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        self.temporal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) 
        
    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        num_var = tf.shape(v)[-2]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        # v : (batch_size, seq_len, num_var, d_model)
   
        v = tf.reshape(v, (batch_size, self.seq_len, -1))
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, temporal_attention_weights = scaled_dot_product_attention(
            q, k, v, self.d_model, self.temporal_mask)

        scaled_attention = tf.reshape(scaled_attention, (batch_size, self.seq_len, num_var, -1))
        
        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, num_var, d_model)

        return output, temporal_attention_weights 
    
class InterpretableMultiHeadAttention(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.d_model, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        scaled_attention_mean = tf.reduce_mean(scaled_attention, axis=2)
        
        concat_attention = tf.reshape(scaled_attention_mean,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    
class SpatialTemporalDecoder(K.layers.Layer):
    def __init__(self, d_model, dr, num_heads):
        super(SpatialTemporalDecoder, self).__init__()
        self.d_model = d_model
        self.dr = dr
        
        self.lstm_future = K.layers.LSTM(d_model,
                        return_sequences=True,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        recurrent_dropout=0,
                        unroll=False,
                        use_bias=True)   
        
        self.glu_and_layer_norm1= GLULN(d_model)
        
        self.imha = InterpretableMultiHeadAttention(d_model, num_heads=num_heads)
        
        self.glu_and_layer_norm2 = GLULN(d_model)
                
    def call(self, vsn_observed, vsn_future):
        time_step = tf.shape(vsn_observed)[1] + tf.shape(vsn_future)[1]
        
        future_lstm = self.lstm_future(vsn_future, initial_state=[vsn_observed[:, -1, :], vsn_observed[:, -1, :]])
        
        lstm_hidden = tf.concat([future_lstm, vsn_observed], axis=1)    
        input_vsn = tf.concat([vsn_future, vsn_observed], axis=1)
        
        glu_phi_list = [] 
        
        for j in range(time_step):
            tmp_phi_t = self.glu_and_layer_norm1(lstm_hidden[:, j, :], input_vsn[:, j, :])
            glu_phi_list.append(tf.expand_dims(tmp_phi_t, axis=1))

        glu_phi = tf.concat(glu_phi_list, axis=1)

        B, _ = self.imha(glu_phi, glu_phi, glu_phi) # imha output, weights

        glu_delta_list = [] 

        for j in range(time_step):
            tmp_delta_t = self.glu_and_layer_norm2(B[:, j, :], glu_phi[:, j, :])
            glu_delta_list.append(tf.expand_dims(tmp_delta_t, axis=1))

        glu_delta = tf.concat(glu_delta_list, axis=1)

        return glu_delta, glu_phi
    
class PointWiseFeedForward(K.layers.Layer):
    def __init__(self, d_model, dr):
        super(PointWiseFeedForward, self).__init__()
        self.grn = GatedResidualNetwork(d_model, dr)
        self.glu_and_layer_norm = GLULN(d_model)

        
    def call(self, delta, phi):
        time_step = tf.shape(delta)[1]
        
        grn_varphi_list = []

        for t in range(time_step):
            tmp_grn_varphi = self.grn(delta[:, t, :])
            grn_varphi_list.append(tf.expand_dims(tmp_grn_varphi, axis=1))

        grn_varphi = tf.concat(grn_varphi_list, axis=1)
        
        varphi_tilde_list = []
        
        for t in range(time_step):
            tmp_varphi_tilde_list = self.glu_and_layer_norm(grn_varphi[:, t, :], phi[:, t, :])
            varphi_tilde_list.append(tf.expand_dims(tmp_varphi_tilde_list, axis=1))
            
        varphi = tf.concat(varphi_tilde_list, axis=1)
            
        return varphi

class TargetFeatureLayer(K.layers.Layer):
    def __init__(self, d_model, num_target):
        super(TargetFeatureLayer, self).__init__()
        self.num_target = num_target
        self.target_feature_dense = [K.layers.Dense(d_model) for _ in range(num_target)]

    def call(self, varphi):
        target_feature_list = []
        for i in range(self.num_target):
            tmp_target_feature = self.target_feature_dense[i](varphi)
            target_feature_list.append(tf.expand_dims(tmp_target_feature, -2))

        return tf.concat(target_feature_list, axis=-2)

class QuantileOutput(K.layers.Layer):
    def __init__(self, tau, quantile):
        super(QuantileOutput, self).__init__()
        self.tau = tau
        self.quantile = quantile
        self.quantile_dense = [K.layers.Dense(1) for _ in range(len(quantile))]

    def call(self, varphi):
        total_output_list = []
        for j in range(len(self.quantile)):
            tmp_quantile_list = []
            for t in range(self.tau):
                tmp_quantile = self.quantile_dense[j](varphi[:, -self.tau + t, ...])
                tmp_quantile_list.append(tf.expand_dims(tmp_quantile, axis=1))
            total_output_list.append(tf.concat(tmp_quantile_list, axis=1))

        return tf.concat(total_output_list, axis=-1)
    