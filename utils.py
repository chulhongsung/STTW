import numpy as np
import pandas as pd

def generate_ts_data(df, label_df, input_seq_len=144, tau=48):
    
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
            label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]

        conti_input_list.append(conti_input)
        cate_input_list.append(cate_input)
        future_input_list.append(future_input)
        label_list.append(label)
    
    total_conti_input = np.concatenate(conti_input_list, axis=0)
    total_cate_input = np.concatenate(cate_input_list, axis=0)
    total_future_input = np.concatenate(future_input_list, axis=0)
    total_label = np.concatenate(label_list, axis=0)
    
    return total_conti_input, total_cate_input, total_future_input, total_label