import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from layer import *

class SpatialTemporalTransformer(K.models.Model):
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        d_input,
        seq_len,
        num_target,
        tau,
        quantile,
        num_heads,
        dr
    ):
        super(SpatialTemporalTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding)
        self.cate_dims = cate_dims
        self.catfe = [CateFeatureEmbedding(d_embedding, x) for x in cate_dims]
        self.ssa = SpatialStructureAttention(d_model, num_heads, spatial_structure)
        self.vsn1 = VariableSelectionNetwork(d_model, d_input, dr)
        self.vsn2 = VariableSelectionNetwork(d_model, d_input, dr)
        self.vsn3 = VariableSelectionNetwork(d_model, len(self.cat_dims), dr)
        self.tsa = TemporalStructureAttention(d_model, num_heads, seq_len)
        self.std = SpatialTemporalDecoder(d_model, dr, num_heads)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_target)
        self.qo = QuantileOutput(tau, quantile)
        
    @tf.function
    def call(self, conti_input, cate_input, future_input):
        
        ### Input Feature Embedding
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_con_var, d_embedding)
        
        tmp_cate_list = [] 

        for k in range(len(self.cate_dims)):
            tmp_cate_input_ = self.catfe[k](cate_input[:, :, k])
            tmp_cate_list.append(tmp_cate_input_)

        cate_embedding = tf.concat(tmp_cate_list, axis=-2) # (batch_size, seq_len, num_cate_var, d_embedding)

        obs_feature = tf.concat([confe_output, cate_embedding], axis=-2) # (batch_size, seq_len, num_var, d_embedding)

        ### Spatio-Temporal Feature Learning
        ssa_result, _ = self.ssa(obs_feature,obs_feature,obs_feature) # (batch_size, seq_len, num_var, d_model), (batch_size, seq_len, num_heads, num_var, num_var) 

        vsn_output = self.vsn1(ssa_result) # (batch_size, seq_len, d_model)
        tsa_output, _ = self.tsa(vsn_output, vsn_output, ssa_result) # (batch_size, seq_len, num_var, d_model)
        
        vsn_output2 = self.vsn2(tsa_output) # (batch_size, seq_len, d_model)
        
        tmp_future_list = [] 
        
        for k in range(len(self.cate_dims)):
            tmp_future_input_ = self.catfe[k](future_input[:, :, k])
            tmp_future_list.append(tmp_future_input_)

        future_embedding = tf.concat(tmp_future_list, axis=-2) # (batch_size, tau, num_cate_var, d_embedding)
        
        vsn_future_output = self.vsn3(future_embedding) # (batch_size, tau, d_model)
        
        std_delta, std_phi = self.std(vsn_output2, vsn_future_output) # (batch_size, seq_len + tau, d_model), (batch_size, seq_len + tau, d_model)
        
        varphi = self.pwff(std_delta, std_phi) # (batch_size, seq_len + tau, d_model)
        
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(varphi) # (batch_size, seq_len + tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile_len)
        
        return output # (batch_size, tau, num_target, quantile_len)