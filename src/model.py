import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from src.layer import *

class SpatialTemporalTransformer(K.models.Model):
    """
    Spatio-Temporal Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        d_input,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        num_heads,
        dr
    ):
        super(SpatialTemporalTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.cate_dims = cate_dims
        self.catfe = [CateFeatureEmbedding(d_embedding, x) for x in cate_dims]
        self.ssa1 = SpatialStructureAttention(d_model, num_heads, spatial_structure)
        self.vsn1 = VariableSelectionNetwork(d_model, d_input, dr)
        
        self.ssa2 = SpatialStructureAttention(d_model, num_heads, spatial_structure)
        
        self.vsn2 = VariableSelectionNetwork(d_model, d_input, dr)
        self.vsn3 = VariableSelectionNetwork(d_model, len(self.cate_dims), dr)
        self.tsa = TemporalStructureAttention(d_model, num_heads, seq_len)
        self.std = SpatialTemporalDecoder(d_model, dr, num_heads)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def call(self, conti_input, cate_input, future_input):
        
        ### Input Feature Embedding
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_con_var, d_embedding)
        
        tmp_cate_list = [] 

        for k in range(len(self.cate_dims)):
            tmp_cate_input_ = self.catfe[k](cate_input[:, :, k])
            tmp_cate_list.append(tmp_cate_input_)

        cate_output = tf.concat(tmp_cate_list, axis=-2) # (batch_size, seq_len, num_cate_var, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = tf.concat([cate_output, confe_output], axis=-2) # (batch_size, seq_len, num_var, d_embedding)

        ssa_result1, ssa_aw = self.ssa1(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_var, d_embedding), (batch_size, seq_len, num_var, num_var) 

        vsn_output1 = self.vsn1(ssa_result1) # (batch_size, seq_len, d_embedding)
                        
        tsa_output1, tsa_aw = self.tsa(vsn_output1, vsn_output1, ssa_result1) # (batch_size, seq_len, num_var, d_model), (batch_size, seq_len, seq_len)
        
        ssa_result2, _ = self.ssa2(tsa_output1, tsa_output1, tsa_output1) # (batch_size, seq_len, num_var, d_model), (batch_size, seq_len, num_heads, num_var, num_var) 

        vsn_output2 = self.vsn2(ssa_result2) # (batch_size, seq_len, d_model)
        
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
        
        return output, ssa_aw[0:1, 100:124, ...], tsa_aw[0:1, ...] # (batch_size, tau, num_target, quantile_len)