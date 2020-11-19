import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sqlnet.model.modules.net_utils import run_lstm, col_name_encode

class OrdPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num):
        super(OrdPredictor, self).__init__()
        self.max_tok_num = max_tok_num
        self.ord_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        print("Using column attention on orderby predicting")
        self.ord_att = nn.Linear(N_h, N_h)

        self.ord_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.ord_out_K = nn.Linear(N_h, N_h)
        self.ord_out_col = nn.Linear(N_h, N_h)
        self.ord_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        # Based on number of ordections to predict ordect-column
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.ord_col_name_enc) # [bs, col_num, hid]
        h_enc, _ = run_lstm(self.ord_lstm, x_emb_var, x_len) # [bs, seq_len, hid]

        att_val = torch.bmm(e_col, self.ord_att(h_enc).transpose(1, 2)) # [bs, col_num, seq_len]
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, :, num:] = -100
        att = self.softmax(att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_ord_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)

        ord_score = self.ord_out( self.ord_out_K(K_ord_expand) + self.ord_out_col(e_col) ).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                ord_score[idx, num:] = -100

        return ord_score
