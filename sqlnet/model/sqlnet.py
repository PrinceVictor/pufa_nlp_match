import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sqlnet.model.modules.word_embedding import WordEmbedding
from sqlnet.model.modules.aggregator_predict import AggPredictor
from sqlnet.model.modules.selection_predict import SelPredictor
from sqlnet.model.modules.sqlnet_condition_predict import SQLNetCondPredictor
from sqlnet.model.modules.select_number import SelNumPredictor
from sqlnet.model.modules.relation import RelationPredictor
from sqlnet.model.modules.orderby_sort_predictor import OrdSortPredictor
from sqlnet.model.modules.orderby_predict import OrdPredictor
from sqlnet.model.modules.groupby_predict import GrpPredictor
from sqlnet.model.modules.groupby_exist import GrpExistPredictor
from sqlnet.model.modules.limit_number import LimNumPredictor
from sqlnet.model.modules.having_predict import HavingPredictor


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
                 gpu=False):
        super(SQLNet, self).__init__()
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'OR', '==', '>', '<', '!=', '<BEG>', '>=', '<=', 'LIKE', 'IN',
                        'NOT IN']
        self.COND_OPS = ['>', '<', '==', '!=', '>=', '<=', 'LIKE']

        # Word embedding
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu, self.SQL_TOK, our_model=True)

        # Predict the number of selected columns
        self.sel_num = SelNumPredictor(N_word, N_h, N_depth)

        # Predict which columns are selected
        self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num)

        # Predict aggregation functions of corresponding selected columns
        self.agg_pred = AggPredictor(N_word, N_h, N_depth)

        # Predict number of conditions, condition columns, condition operations and condition values
        self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)

        # Predict condition relationship, like 'and', 'or'
        self.where_rela_pred = RelationPredictor(N_word, N_h, N_depth)

        # Predict how order by column sorted, like 'desc', 'asc'
        self.orderby_sort_pred = OrdSortPredictor(N_word, N_h, N_depth)

        # Predict which columns are selected
        self.orderby_pred = OrdPredictor(N_word, N_h, N_depth, self.max_tok_num)

        # Predict aggregation functions of corresponding orderby columns
        self.orderby_agg_pred = AggPredictor(N_word, N_h, N_depth)

        # Predict whether groupby is used
        self.grpby_exist_pred = GrpExistPredictor(N_word, N_h, N_depth)

        # Predict which column is selected
        self.grpby_pred = GrpPredictor(N_word, N_h, N_depth, self.max_tok_num)

        # Predict the number of limit rows
        self.lim_num = LimNumPredictor(N_word, N_h, N_depth)

        # Predict having
        self.having_pred = HavingPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)

        # Predict aggregation functions of corresponding having columns
        self.having_agg_pred = AggPredictor(N_word, N_h, N_depth)

        # Predict condition relationship, like 'and', 'or'
        self.having_rela_pred = RelationPredictor(N_word, N_h, N_depth)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()

    def generate_gt_where_seq_test(self, q, gt_cond_seq):
        ret_seq = []
        for cur_q, ans in zip(q, gt_cond_seq):
            temp_q = u"".join(cur_q)
            cur_q = [u'<BEG>'] + cur_q + [u'<END>']
            record = []
            record_cond = []
            for cond in ans:
                if cond[2] not in temp_q:
                    record.append((False, cond[2]))
                else:
                    record.append((True, cond[2]))
            for idx, item in enumerate(record):
                temp_ret_seq = []
                if item[0]:
                    temp_ret_seq.append(0)
                    temp_ret_seq.extend(
                        list(
                            range(temp_q.index(item[1]) + 1, temp_q.index(item[1]) + len(item[1]) + 1)))
                    temp_ret_seq.append(len(cur_q) - 1)
                else:
                    temp_ret_seq.append([0, len(cur_q) - 1])
                record_cond.append(temp_ret_seq)
            ret_seq.append(record_cond)
        return ret_seq

    def generate_gt_having_seq_test(self, q, gt_having_seq):
        ret_seq = []
        for cur_q, ans in zip(q, gt_having_seq):
            temp_q = u"".join(cur_q)
            cur_q = [u'<BEG>'] + cur_q + [u'<END>']
            record = []
            record_cond = []
            for cond in ans:
                if cond[2] not in temp_q:
                    record.append((False, cond[2]))
                else:
                    record.append((True, cond[2]))
            for idx, item in enumerate(record):
                temp_ret_seq = []
                if item[0]:
                    temp_ret_seq.append(0)
                    temp_ret_seq.extend(
                        list(
                            range(temp_q.index(item[1]) + 1, temp_q.index(item[1]) + len(item[1]) + 1)))
                    temp_ret_seq.append(len(cur_q) - 1)
                else:
                    temp_ret_seq.append([0, len(cur_q) - 1])
                record_cond.append(temp_ret_seq)
            ret_seq.append(record_cond)
        return ret_seq

    def forward(self, q, col, col_num, gt_where=None, gt_cond=None, reinforce=False, gt_sel=None, gt_sel_num=None,
                gt_orderby=None, gt_orderby_num=None, gt_having=None, gt_having_num=None, gt_having_cond=None,
                gt_having_agg=None):
        B = len(q)

        sel_num_score = None
        agg_score = None
        sel_score = None
        cond_score = None
        # Predict aggregator
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
        col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)
        sel_num_score = self.sel_num(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        # x_emb_var: embedding of each question
        # x_len: length of each question
        # col_inp_var: embedding of each header
        # col_name_len: length of each header
        # col_len: number of headers in each table, array type
        # col_num: number of headers in each table, list type
        if gt_sel_num:
            pr_sel_num = gt_sel_num
        else:
            pr_sel_num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
        sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        if gt_sel:
            pr_sel = gt_sel
        else:
            num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
            sel = sel_score.data.cpu().numpy()
            pr_sel = [list(np.argsort(-sel[b])[:num[b]]) for b in range(len(num))]
        agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_sel=pr_sel,
                                  gt_sel_num=pr_sel_num)

        where_rela_score = self.where_rela_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        order_sort_score = self.orderby_sort_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        if gt_orderby_num:
            pr_orderby_num = gt_orderby_num
        else:
            pr_orderby_num = np.argmax(order_sort_score.data.cpu().numpy(), axis=1)
            pr_orderby_num = np.array(list(map(lambda x: (x + 1) // 2, pr_orderby_num)))

        orderby_score = self.orderby_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        if gt_orderby:
            pr_orderby = gt_orderby
        else:
            num = np.argmax(order_sort_score.data.cpu().numpy(), axis=1)
            num = np.array(list(map(lambda x: (x + 1) // 2, num)))
            orderby = orderby_score.data.cpu().numpy()
            pr_orderby = [list(np.argsort(-orderby[b])[:num[b]]) for b in range(len(num))]

        orderby_agg_score = self.orderby_agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num,
                                                  gt_sel=pr_orderby, gt_sel_num=pr_orderby_num)
        grpby_exist_score = self.grpby_exist_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        grpby_score = self.grpby_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_where,
                                    gt_cond, reinforce=reinforce)
        lim_num_score = self.lim_num(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        having_score = self.having_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_having,
                                        gt_having_cond, reinforce=reinforce)

        having_num_sore = having_score[0]
        if gt_having_num:
            pr_having_num = gt_having_num
        else:
            pr_having_num = np.argmax(having_num_sore.data.cpu().numpy(), axis=1)

        having_col_score = having_score[1]
        if gt_having_agg:
            pr_having_agg = gt_having_agg
        else:
            num = np.argmax(having_num_sore.data.cpu().numpy(), axis=1)
            having_col = having_col_score.data.cpu().numpy()
            pr_having_agg = [list(np.argsort(-having_col[b])[:num[b]]) for b in range(len(num))]

        having_agg_score = self.having_agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num,
                                                gt_sel=pr_having_agg, gt_sel_num=pr_having_num)
        having_rela_score = self.where_rela_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        return (sel_num_score, sel_score, agg_score, cond_score, where_rela_score, order_sort_score, orderby_score,
                orderby_agg_score, grpby_exist_score, grpby_score, lim_num_score, having_score, having_agg_score,
                having_rela_score)

    def loss(self, score, truth_num, gt_where, gt_having):
        sel_num_score, sel_score, agg_score, cond_score, where_rela_score, order_sort_score, orderby_score, orderby_agg_score, grpby_exist_score, grpby_score, lim_num_score, having_score, having_agg_score, having_rela_score = score

        B = len(truth_num)
        loss = 0

        # Evaluate select number
        sel_num_truth = list(map(lambda x: x[0], truth_num))
        sel_num_truth = torch.from_numpy(np.array(sel_num_truth))
        if self.gpu:
            sel_num_truth = Variable(sel_num_truth.cuda())
        else:
            sel_num_truth = Variable(sel_num_truth)
        loss += self.CE(sel_num_score, sel_num_truth)

        # Evaluate select column
        T = len(sel_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][list(truth_num[b][1])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            sel_col_truth_var = Variable(data.cuda())
        else:
            sel_col_truth_var = Variable(data)
        sigm = nn.Sigmoid()
        sel_col_prob = sigm(sel_score)
        bce_loss = -torch.mean(
            3 * (sel_col_truth_var * torch.log(sel_col_prob + 1e-10)) +
            (1 - sel_col_truth_var) * torch.log(1 - sel_col_prob + 1e-10)
        )
        loss += bce_loss

        # Evaluate select aggregation
        for b in range(len(truth_num)):
            data = torch.from_numpy(np.array(truth_num[b][2]))
            if self.gpu:
                sel_agg_truth_var = Variable(data.cuda())
            else:
                sel_agg_truth_var = Variable(data)
            sel_agg_pred = agg_score[b, :len(truth_num[b][1])]
            loss += (self.CE(sel_agg_pred, sel_agg_truth_var)) / len(truth_num)

        cond_num_score, cond_col_score, cond_op_score, cond_str_score = cond_score

        # Evaluate the number of conditions
        cond_num_truth = list(map(lambda x: x[3], truth_num))
        data = torch.from_numpy(np.array(cond_num_truth))
        if self.gpu:
            try:
                cond_num_truth_var = Variable(data.cuda())
            except:
                print("cond_num_truth_var error")
                print(data)
                exit(0)
        else:
            cond_num_truth_var = Variable(data)
        loss += self.CE(cond_num_score, cond_num_truth_var.long())

        # Evaluate the columns of conditions
        T = len(cond_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][4]) > 0:
                truth_prob[b][list(truth_num[b][4])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            cond_col_truth_var = Variable(data.cuda())
        else:
            cond_col_truth_var = Variable(data)

        sigm = nn.Sigmoid()
        cond_col_prob = sigm(cond_col_score)
        bce_loss = -torch.mean(
            3 * (cond_col_truth_var * torch.log(cond_col_prob + 1e-10)) +
            (1 - cond_col_truth_var) * torch.log(1 - cond_col_prob + 1e-10))
        loss += bce_loss

        # Evaluate the operator of conditions
        for b in range(len(truth_num)):
            if len(truth_num[b][5]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][5]))
            if self.gpu:
                cond_op_truth_var = Variable(data.cuda())
            else:
                cond_op_truth_var = Variable(data)
            cond_op_pred = cond_op_score[b, :len(truth_num[b][5])]
            try:
                loss += (self.CE(cond_op_pred, cond_op_truth_var) / len(truth_num))
            except:
                # print (cond_op_pred)
                # print (cond_op_truth_var)
                exit(0)

        # Evaluate the strings of conditions
        for b in range(len(gt_where)):
            for idx in range(len(gt_where[b])):
                cond_str_truth = gt_where[b][idx]
                if len(cond_str_truth) == 1:
                    continue
                data = torch.from_numpy(np.array(cond_str_truth[1:]))
                if self.gpu:
                    cond_str_truth_var = Variable(data.cuda())
                else:
                    cond_str_truth_var = Variable(data)
                str_end = len(cond_str_truth) - 1
                cond_str_pred = cond_str_score[b, idx, :str_end]
                loss += (self.CE(cond_str_pred, cond_str_truth_var) \
                         / (len(gt_where) * len(gt_where[b])))

        # Evaluate condition relationship, and / or
        where_rela_truth = list(map(lambda x: x[6], truth_num))
        data = torch.from_numpy(np.array(where_rela_truth))
        if self.gpu:
            try:
                where_rela_truth = Variable(data.cuda())
            except:
                print("where_rela_truth error")
                print(data)
                exit(0)
        else:
            where_rela_truth = Variable(data)
        loss += self.CE(where_rela_score, where_rela_truth)

        # Evaluate condition orderby sort, desc / asc
        orderby_sort_truth = list(map(lambda x: x[8], truth_num))
        data = torch.from_numpy(np.array(orderby_sort_truth))
        if self.gpu:
            try:
                orderby_sort_truth = Variable(data.cuda())
            except:
                print("orderby_sort_truth error")
                print(data)
                exit(0)
        else:
            orderby_sort_truth = Variable(data)
        loss += self.CE(order_sort_score, orderby_sort_truth)

        # Evaluate orderby column
        T = len(orderby_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][list(truth_num[b][9])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            orderby_col_truth_var = Variable(data.cuda())
        else:
            orderby_col_truth_var = Variable(data)
        sigm = nn.Sigmoid()
        orderby_col_prob = sigm(orderby_score)
        bce_loss = -torch.mean(
            3 * (orderby_col_truth_var * torch.log(orderby_col_prob + 1e-10)) +
            (1 - orderby_col_truth_var) * torch.log(1 - orderby_col_prob + 1e-10)
        )
        loss += bce_loss

        # Evaluate orderby aggregation
        for b in range(len(truth_num)):
            data = torch.from_numpy(np.array(truth_num[b][10]))
            if data.shape[0] <= 0:
                continue
            if self.gpu:
                orderby_agg_truth_var = Variable(data.cuda())
            else:
                orderby_agg_truth_var = Variable(data)
            orderby_agg_pred = orderby_agg_score[b, :len(truth_num[b][10])]
            loss += (self.CE(orderby_agg_pred, orderby_agg_truth_var.long())) / len(truth_num)

        # Evaluate whether groupby exist
        groupby_exist_truth = list(map(lambda x: x[11], truth_num))
        data = torch.from_numpy(np.array(groupby_exist_truth))
        if self.gpu:
            try:
                groupby_exist_truth = Variable(data.cuda())
            except:
                print("groupby_exist_truth error")
                print(data)
                exit(0)
        else:
            groupby_exist_truth = Variable(data)
        loss += self.CE(grpby_exist_score, groupby_exist_truth)

        # Evaluate groupby column
        T = len(grpby_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][list(truth_num[b][12])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            grpby_col_truth_var = Variable(data.cuda())
        else:
            grpby_col_truth_var = Variable(data)
        sigm = nn.Sigmoid()
        grpby_col_prob = sigm(grpby_score)
        bce_loss = -torch.mean(
            3 * (grpby_col_truth_var * torch.log(grpby_col_prob + 1e-10)) +
            (1 - grpby_col_truth_var) * torch.log(1 - grpby_col_prob + 1e-10)
        )
        loss += bce_loss

        # Evaluate limit number
        lim_num_truth = list(map(lambda x: x[13], truth_num))
        lim_num_truth = torch.from_numpy(np.array(lim_num_truth))
        if self.gpu:
            lim_num_truth = Variable(lim_num_truth.cuda())
        else:
            lim_num_truth = Variable(lim_num_truth)
        loss += self.CE(lim_num_score, lim_num_truth)

        having_num_score, having_col_score, having_op_score, having_str_score = having_score

        # Evaluate whether having exist or not
        having_num_truth = list(map(lambda x: x[14], truth_num))
        data = torch.from_numpy(np.array(having_num_truth))
        if self.gpu:
            try:
                having_num_truth_var = Variable(data.cuda())
            except:
                print("cond_num_truth_var error")
                print(data)
                exit(0)
        else:
            having_num_truth_var = Variable(data)
        loss += self.CE(having_num_score, having_num_truth_var.long())

        # Evaluate the columns of having
        T = len(having_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][15]) > 0:
                truth_prob[b][list(truth_num[b][15])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            having_col_truth_var = Variable(data.cuda())
        else:
            having_col_truth_var = Variable(data)

        sigm = nn.Sigmoid()
        having_col_prob = sigm(having_col_score)
        bce_loss = -torch.mean(
            3 * (having_col_truth_var * torch.log(having_col_prob + 1e-10)) +
            (1 - having_col_truth_var) * torch.log(1 - having_col_prob + 1e-10))
        loss += bce_loss

        # Evaluate the operator of having
        for b in range(len(truth_num)):
            if len(truth_num[b][16]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][16]))
            if self.gpu:
                having_op_truth_var = Variable(data.cuda())
            else:
                having_op_truth_var = Variable(data)
            having_op_pred = having_op_score[b, :len(truth_num[b][16])]
            try:
                loss += (self.CE(having_op_pred, having_op_truth_var) / len(truth_num))
            except:
                # print (cond_op_pred)
                # print (cond_op_truth_var)
                exit(0)

        # Evaluate the strings of having
        for b in range(len(gt_having)):
            for idx in range(len(gt_having[b])):
                having_str_truth = gt_having[b][idx]
                if len(having_str_truth) == 1:
                    continue
                data = torch.from_numpy(np.array(having_str_truth[1:]))
                if self.gpu:
                    having_str_truth_var = Variable(data.cuda())
                else:
                    having_str_truth_var = Variable(data)
                str_end = len(having_str_truth) - 1
                having_str_pred = having_str_score[b, idx, :str_end]
                loss += (self.CE(having_str_pred, having_str_truth_var) \
                         / (len(gt_having) * len(gt_having[b])))

        # Evaluate having aggregation
        for b in range(len(truth_num)):
            data = torch.from_numpy(np.array(truth_num[b][17]))
            if data.shape[0] <= 0:
                continue
            if self.gpu:
                having_agg_truth_var = Variable(data.cuda())
            else:
                having_agg_truth_var = Variable(data)
            having_agg_pred = having_agg_score[b, :len(truth_num[b][17])]
            loss += (self.CE(having_agg_pred, having_agg_truth_var.long())) / len(truth_num)

        # Evaluate condition relationship, and / or
        having_rela_truth = list(map(lambda x: x[18], truth_num))
        data = torch.from_numpy(np.array(having_rela_truth))
        if self.gpu:
            try:
                having_rela_truth = Variable(data.cuda())
            except:
                print("having_rela_truth error")
                print(data)
                exit(0)
        else:
            having_rela_truth = Variable(data)
        loss += self.CE(having_rela_score, having_rela_truth)

        return loss

    def check_acc(self, vis_info, pred_queries, gt_queries):
        tot_err = sel_num_err = agg_err = sel_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
        orderby_sort_err = orderby_err = orderby_agg_err = 0.0
        groupby_exist_err = groupby_err = 0.0
        limit_num_err = 0.0
        having_col_err = having_op_err = having_val_err = having_agg_err = having_num_err = having_rela_err = 0.0
        except_err = union_err = intersect_err = 0.0
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            sel_pred, agg_pred, where_rela_pred, orderby_pred, orderby_sort_pred, orderby_agg_pred, groupby_pred, limit_num_pred, having_pred, having_agg_pred, having_rela_pred = \
                pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op'], pred_qry['orderby'], pred_qry[
                    'orderby_sort'], \
                pred_qry['orderby_agg'], pred_qry['groupby'], pred_qry['limit'], pred_qry['having'], pred_qry[
                    'having_agg'], pred_qry['having_conn_op']
            sel_gt, agg_gt, where_rela_gt, orderby_gt, orderby_sort_gt, orderby_agg_gt, groupby_gt, limit_num_gt, having_gt, having_agg_gt, having_rela_gt = \
                gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op'], gt_qry['orderby'], gt_qry['orderby_sort'], gt_qry[
                    'orderby_agg'], gt_qry['groupby'], gt_qry['limit'], gt_qry['having'], gt_qry['having_agg'], gt_qry[
                    'having_conn_op']

            except_pred, union_pred, intersect_pred = pred_qry['except'], pred_qry['union'], pred_qry['intersect']
            except_gt, union_gt, intersect_gt = gt_qry['except'], gt_qry['union'], gt_qry['intersect']

            if except_pred != except_gt:
                good = False
                except_err += 1

            if union_pred != union_gt:
                good = False
                union_err += 1

            if intersect_pred != intersect_gt:
                good = False
                intersect_err += 1

            if where_rela_gt != where_rela_pred:
                good = False
                cond_rela_err += 1

            if len(sel_pred) != len(sel_gt):
                good = False
                sel_num_err += 1

            if len(groupby_pred) != len(groupby_gt):
                good = False
                groupby_exist_err += 1

            if groupby_pred != groupby_gt:
                good = False
                groupby_err += 1

            if limit_num_pred != limit_num_gt:
                good = False
                limit_num_err += 1

            pred_sel_dict = {k: v for k, v in zip(list(sel_pred), list(agg_pred))}
            gt_sel_dict = {k: v for k, v in zip(sel_gt, agg_gt)}
            if set(sel_pred) != set(sel_gt):
                good = False
                sel_err += 1
            agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
            agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
            if agg_pred != agg_gt:
                good = False
                agg_err += 1

            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['conds']
            if len(cond_pred) != len(cond_gt):
                good = False
                cond_num_err += 1
                cond_col_err += 1
                cond_rela_err += 1
                cond_op_err += 1
                cond_val_err += 1
            else:
                cond_op_pred, cond_op_gt = {}, {}
                cond_val_pred, cond_val_gt = {}, {}
                for p, g in zip(cond_pred, cond_gt):
                    cond_op_pred[p[0]] = p[1]
                    cond_val_pred[p[0]] = p[2]
                    cond_op_gt[g[0]] = g[1]
                    cond_val_gt[g[0]] = g[2]

                if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
                    cond_col_err += 1
                    good = False

                where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
                where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
                if where_op_pred != where_op_gt:
                    cond_op_err += 1
                    good = False

                where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
                where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
                if where_val_pred != where_val_gt:
                    cond_val_err += 1
                    good = False
            if set(orderby_pred) != set(orderby_gt):
                orderby_err += 1
                good = False
            if orderby_sort_pred != orderby_sort_gt:
                orderby_sort_err += 1
                good = False
            if set(orderby_agg_pred) != set(orderby_agg_gt):
                orderby_agg_err += 1
                good = False

            if len(having_pred) != len(having_gt):
                good = False
                having_col_err += 1
                having_op_err += 1
                having_val_err += 1
                having_agg_err += 1
                having_num_err += 1
                having_rela_err += 1
            else:
                having_op_pred, having_op_gt = {}, {}
                having_val_pred, having_val_gt = {}, {}
                for p, g in zip(having_pred, having_gt):
                    having_op_pred[p[0]] = p[1]
                    having_val_pred[p[0]] = p[2]
                    having_op_gt[g[0]] = g[1]
                    having_val_gt[g[0]] = g[2]

                if set(having_op_pred.keys()) != set(having_op_gt.keys()):
                    having_col_err += 1
                    good = False

                having_op_pred = [having_op_pred[x] for x in sorted(having_op_pred.keys())]
                having_op_gt = [having_op_gt[x] for x in sorted(having_op_gt.keys())]
                if having_op_pred != having_op_gt:
                    having_op_err += 1
                    good = False

                having_val_pred = [having_val_pred[x] for x in sorted(having_val_pred.keys())]
                having_val_gt = [having_val_gt[x] for x in sorted(having_val_gt.keys())]
                if having_val_pred != having_val_gt:
                    having_val_err += 1
                    good = False

                if having_agg_pred != having_agg_gt:
                    good = False
                    having_agg_err += 1

                if having_rela_gt != having_rela_pred:
                    good = False
                    having_rela_err += 1

            if not good:
                tot_err += 1

        return np.array((sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err,
                         cond_rela_err, orderby_err, orderby_sort_err, orderby_agg_err, groupby_exist_err, groupby_err,
                         limit_num_err, having_col_err, having_op_err, having_val_err, having_agg_err, having_num_err,
                         having_rela_err, except_err, union_err, intersect_err)), tot_err

    def gen_query(self, score, q, col, raw_q, reinforce=False, verbose=False):
        """
        :param score:
        :param q: token-questions
        :param col: token-headers
        :param raw_q: original question sequence
        :return:
        """

        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str  # .lower()
            special = {'-LRB-': '(',
                       '-RRB-': ')',
                       '-LSB-': '[',
                       '-RSB-': ']',
                       '``': '"',
                       '\'\'': '"',
                       '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear
                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret  # + ' '
                # elif tok[0] not in alphabet:
                #     pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret  # + ' '
                ret = ret + tok
            return ret.strip()

        sel_num_score, sel_score, agg_score, cond_score, where_rela_score, orderby_sort_score, orderby_score, orderby_agg_score, grpby_exist_score, grpby_score, lim_num_score, having_score, having_agg_score, having_rela_score = score
        # [64,4,6], [64,14], ..., [64,4]
        sel_num_score = sel_num_score.data.cpu().numpy()
        sel_score = sel_score.data.cpu().numpy()
        agg_score = agg_score.data.cpu().numpy()
        where_rela_score = where_rela_score.data.cpu().numpy()
        orderby_sort_score = orderby_sort_score.data.cpu().numpy()
        orderby_score = orderby_score.data.cpu().numpy()
        orderby_agg_score = orderby_agg_score.data.cpu().numpy()
        grpby_exist_score = grpby_exist_score.data.cpu().numpy()
        grpby_score = grpby_score.data.cpu().numpy()
        lim_num_score = lim_num_score.data.cpu().numpy()
        having_agg_score = having_agg_score.data.cpu().numpy()
        having_rela_score = having_rela_score.data.cpu().numpy()

        ret_queries = []
        sql_queries = []
        B = len(agg_score)
        cond_num_score, cond_col_score, cond_op_score, cond_str_score = \
            [x.data.cpu().numpy() for x in cond_score]
        having_num_score, having_col_score, having_op_score, having_str_score = \
            [x.data.cpu().numpy() for x in having_score]
        for b in range(B):
            cur_query = {}
            cur_query['sel'] = []
            cur_query['agg'] = []
            sel_num = np.argmax(sel_num_score[b])
            max_col_idxes = np.argsort(-sel_score[b])[:sel_num]
            # find the most-probable columns' indexes
            max_agg_idxes = np.argsort(-agg_score[b])[:sel_num]
            cur_query['sel'].extend([int(i) for i in max_col_idxes])
            cur_query['agg'].extend([i[0] for i in max_agg_idxes])
            cur_query['cond_conn_op'] = np.argmax(where_rela_score[b])
            cur_query['having_conn_op'] = np.argmax(having_rela_score[b])
            cur_query['orderby_sort'] = np.argmax(orderby_sort_score[b])
            cur_query['orderby'] = []
            cur_query['orderby_agg'] = []
            if np.argmax(orderby_sort_score[b]) != 0:
                cur_query['orderby'].append(np.argmax(orderby_score[b]))
                cur_query['orderby_agg'].append(np.argsort(-orderby_agg_score[b])[0][0])

            cur_query['groupby'] = []
            if np.argmax(grpby_exist_score[b]) != 0:
                cur_query['groupby'].append(np.argmax(grpby_score[b]))

            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(max_idxes[idx])  # where-col
                cur_cond.append(np.argmax(cond_op_score[b][idx]))  # where-op
                cur_cond_str_toks = []
                for str_score in cond_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]

                    if str_val == '<END>':
                        break
                    if [str_tok, str_val] not in cur_cond_str_toks:
                        cur_cond_str_toks.append([str_tok, str_val])
                sorted(cur_cond_str_toks)
                cur_cond_str_toks_new = [_[1] for _ in cur_cond_str_toks]

                cur_cond.append(merge_tokens(list(cur_cond_str_toks_new), raw_q[b]))
                cur_query['conds'].append(cur_cond)

            cur_query['having'] = []
            having_num = np.argmax(having_num_score[b])
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            max_idxes = np.argsort(-having_col_score[b])[:having_num]
            for idx in range(having_num):
                cur_having = []
                cur_having.append(max_idxes[idx])  # having-col
                cur_having.append(np.argmax(having_op_score[b][idx]))  # having-op
                cur_having_str_toks = []
                for str_score in having_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]

                    if str_val == '<END>':
                        break
                    if [str_tok, str_val] not in cur_having_str_toks:
                        cur_having_str_toks.append([str_tok, str_val])
                sorted(cur_having_str_toks)
                cur_having_str_toks_new = [_[1] for _ in cur_having_str_toks]

                cur_having.append(merge_tokens(list(cur_having_str_toks_new), raw_q[b]))
                cur_query['having'].append(cur_having)

            cur_query['having_agg'] = []
            max_having_agg_idxes = np.argsort(-having_agg_score[b])[:having_num]
            cur_query['having_agg'].extend([i[0] for i in max_having_agg_idxes])

            cur_query['limit'] = np.argmax(lim_num_score[b])
            cur_query['except'] = {}
            cur_query['union'] = {}
            cur_query['intersect'] = {}

            agg_dict = {0: '', 1: "AVG", 2: "COUNT", 3: "SUM", 4: "MIN", 5: "MAX"}
            rel_dict = {0: '', 1: 'AND', 2: 'OR'}
            cal_dict = {0: '', 1: "+", 2: "-", 3: "*", 4: "/"}
            judge_dict = {0: '', 1: "!=", 2: "==", 3: ">", 4: "<", 5: ">=", 6: "<=", 7: "LIKE", 8: "IN", 9: "NOT IN"}
            orderby_sort_dict = {0: '', 1: 'DESC', 2: 'ASC'}
            sel_cols = cur_query['sel']
            sel_aggs = cur_query['agg']

            sub1 = ''
            tmp_col = ''
            tmp_where_col = ''
            for i, sel_col in enumerate(sel_cols):
                tmp_col = ''.join(col[b][sel_col])
                tmp_col_name = ''.join(tmp_col.split('.')[1:]) if len(tmp_col) > 1 else ''.join(tmp_col.split('.')[0])
                if tmp_col_name != '*':
                    tmp_col_name = '"' + tmp_col_name + '"'
                if sel_aggs[i] != 0:
                    tmp_col_name = '(' + tmp_col_name + ')'
                tmp = "{}{},".format(agg_dict[sel_aggs[i]], tmp_col_name)

                sub1 += tmp

            sub1 = sub1.strip(',')
            rel = rel_dict[cur_query['cond_conn_op']]
            if rel != 'OR':
                rel = 'AND'

            sub2 = ''
            for where_sub in cur_query['conds']:
                tmp_where_col = ''.join(col[b][where_sub[0]])
                where_col = ''.join(tmp_where_col.split('.')[1:]) if len(tmp_where_col) > 1 else \
                    ''.join(tmp_where_col.split('.')[0])
                try:
                    float(where_sub[2])
                    tmp3 = " \"{}\" {} {} ".format(where_col, judge_dict[where_sub[1]], where_sub[2])
                except:
                    tmp3 = " \"{}\" {} '{}' ".format(where_col, judge_dict[where_sub[1]], where_sub[2])

                sub2 += tmp3
                sub2 += rel
            sub2 = sub2.strip(rel)

            sub3 = ''
            if len(cur_query['groupby']) != 0:
                groupby_col = '"' + ''.join(col[b][cur_query['groupby'][0]]).split('.')[-1] + '"'
                sub3 = "GROUP BY {}".format(groupby_col)

            sub4 = ''
            if cur_query['having'] != []:
                sub4 = 'HAVING'
            rel_having = rel_dict[cur_query['having_conn_op']]
            if rel_having != 'OR':
                rel_having = 'AND'
            for idx, having_sub in enumerate(cur_query['having']):
                tmp_having_col = '"' + ''.join(col[b][having_sub[0]]).split('.')[-1] + '"'
                # having_agg = cur_query['having_agg'][idx]
                if len(cur_query['having_agg']) > 0 and cur_query['having_agg'][idx] != 0:
                    tmp_having_col = agg_dict[cur_query['having_agg'][idx]] + '(' + tmp_having_col + ')'
                try:
                    float(cur_query['having'][idx][2])
                    tmp4 = " {} {} {} ".format(tmp_having_col, judge_dict[cur_query['having'][idx][1]],
                                               cur_query['having'][idx][2])
                except:
                    tmp4 = " {} {} '{}' ".format(tmp_having_col, judge_dict[cur_query['having'][idx][1]],
                                                 cur_query['having'][idx][2])
                sub4 += tmp4
                sub4 += rel_having
            sub4 = sub4.strip(rel_having)

            sub5 = ''
            if cur_query['orderby_sort'] != 0:
                orderby_col = '"' + ''.join(col[b][cur_query['orderby'][0]]).split('.')[-1] + '"'
                if agg_dict[cur_query['orderby_agg'][0]] != '':
                    orderby_col = '(' + orderby_col + ')'

                sub5 = "ORDER BY {}{} {} ".format(agg_dict[cur_query['orderby_agg'][0]], orderby_col,
                                                  orderby_sort_dict[cur_query['orderby_sort']])

            table_name = ''.join(tmp_col.split('.')[0]) if len(tmp_col) > 1 else \
                ''.join(tmp_where_col.split('.')[0])
            if table_name in ["*",""]:
                table_name = ''.join(''.join(col[b][1]).split('.')[0])
            if sub2 != '':
                sql_query = "SELECT {} FROM \"{}\" WHERE{}{}{}{}".format(sub1, table_name, sub2, sub3, sub4, sub5)
            else:
                sql_query = "SELECT {} FROM \"{}\" {}{}{}".format(sub1, table_name, sub3, sub4, sub5)

            if cur_query['limit'] > 0:
                sql_query = sql_query + "LIMIT {}".format(cur_query['limit'])
            #print(cur_query)
            #print(sql_query)

            ret_queries.append(cur_query)
            sql_queries.append(sql_query)

        return ret_queries, sql_queries
