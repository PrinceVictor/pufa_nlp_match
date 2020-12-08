import json
import numpy as np
from tqdm import tqdm
import math
from sqlnet.lib.strPreprocess import *
from sqlnet.model.sqlbert import SQLBert
from fuzzywuzzy import process
from fuzzywuzzy.utils import StringProcessor
from functools import lru_cache
from sqlnet.lib.diff2 import extact_sort

@lru_cache(None)
def my_scorer(t, c):
    return (1 - abs(len(t) - len(c)) / max(len(t), len(c))) * process.default_scorer(t, c)


def my_process(s):
    """Process string by
        -- removing all but letters and numbers
        -- trim whitespace
        -- force to lower case
        if force_ascii == True, force convert to ascii"""
    # Force into lowercase.
    string_out = StringProcessor.to_lower_case(s)
    # Remove leading and trailing whitespaces.
    string_out = StringProcessor.strip(string_out)
    return string_out

def pos_in_tokens(target_str, tokens, type = None, header = None):
    if not tokens:
        return -1, -1
    tlen = len(target_str)
    copy_target_str = target_str
    q = ''.join(tokens).replace('##', '')
    header = ''.join(header).replace('##','').replace('[UNK]','')
    ngrams = []
    for l in range(max(1, tlen - 25), min(tlen + 5, len(tokens))):
        ngrams.append(l)
    candidates = {}
    unit_flag = 0
    tback_flag = 0
    unit_r = 0
    if type =='real':
        units = re.findall(r'[(（-](.*)',str(header))
        if units:
            unit = units[0]
            #unit_keys = re.findall(r'[百千万亿]{1,}',str(header))
            unit_keys = re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', unit)
            unit_other = re.findall(r'元|米|平|套|枚|册|张|辆|个|股|户|m²|亩|人', unit)
            if unit_keys:
                unit_flag = 1        #col中有[万|百万|千万|亿]单位，
                unit_key = unit_keys[0]
                v, unit_r = chinese_to_digits(unit_key)
                #print('--unit--',unit_key, target_str)
                target_str = target_str + unit_key
                target_str = strPreProcess(target_str)
                target_str = unit_convert(target_str)
                #print('--target_str--', target_str, header)
            elif unit_other:
                unit_flag = 2    #col中有[元|米|平] 单位为个数
            else:
                unit_flag = 3    # 无单位，可能为个数，可能与ques中单位相同
    for l in ngrams:
        cur_idx = 0
        while cur_idx <= len(tokens) - l:
            cur_str = []
            st, ed = cur_idx, cur_idx
            i = st
            while i != len(tokens) and len(cur_str) < l:
                cur_tok = tokens[i]
                cur_str.append(cur_tok)
                i += 1
                ed = i
            cur_str = ''.join(cur_str)
            if '##' in cur_str :
                cur_str = cur_str.replace('##', '')
            if '[UNK]' in cur_str :
                cur_str = cur_str.replace('[UNK]', '')
            if '-' in cur_str :
                cur_str = cur_str.replace('-', '')

            if unit_flag == 1:
                if cur_str == target_str: #ques 无单位 默认为个数 target_str为unit_convert()后的
                    cur_str = str(int(cur_str)/unit_r)
                    unit_flag = 0 #target_str回到初始状态，
                    tback_flag = 0
                # elif cur_str == copy_target_str: #ques 无单位 默认与target_str 相同
                #     tback_flag = 1 #标志位 默认与target_str 单位相同
                else:
                    cur_str = unit_convert(cur_str)

            elif unit_flag == 2:
                cur_str = unit_convert(cur_str)
            elif unit_flag == 3:
                if unit_convert(cur_str) == target_str:
                    cur_str = unit_convert(cur_str)
            if type == 'text':
                for item in list(thesaurus_dic.keys()):
                    if item in cur_str:
                        cur_str_the = re.sub(item,thesaurus_dic[item],cur_str)
                        candidates[cur_str_the] = (st, ed)
            candidates[cur_str] = (st, ed)
            cur_idx += 1
    # if tback_flag:
    #     target_str = copy_target_str

    if list(candidates.keys()) is None or len(list(candidates.keys())) == 0:
        # print('-----testnone----',target_str, tokens,ngrams)
        return -1, -1

    target_str = str(target_str).replace('-', '')
    resultsf = process.extract(target_str, list(candidates.keys()), limit=10, processor=my_process, scorer=my_scorer)
    results = extact_sort(target_str, list(candidates.keys()), limit=10)
    if not results or not resultsf:
        return -1, -1
    dchosen, dcscore = results[0]
    fchosen, fcscore = resultsf[0]
    if fcscore > dcscore:
        cscore = fcscore
        chosen = fchosen
    else:
        cscore = dcscore
        chosen = dchosen

    if cscore !=100:
        pass
        #q = ''.join(tokens).replace('##','')
        #score = '%d'%(cscore)
        #with open("F:\\天池比赛\\nl2sql_test_20190618\\log3.txt", "a", encoding='utf-8') as fw:
            #fw.write(str(chosen + '-----' + target_str + '---'+score +'--'+ q +'\n'+'\n'))

    if cscore <= 50:
        q = ''.join(tokens).replace('##','')
        score = '%d'%(cscore)
        #with open("F:\\天池比赛\\nl2sql_test_20190618\\log3.txt", "a", encoding='utf-8') as fw:
            #fw.write(str(type + '  '+ header + ' ** '+chosen + '-----' + target_str + '---'+score +'--'+ q +'\n'+'\n'))
        #return -1, -1
    return candidates[chosen]
    #return cscore, chosen
    
def filter(total_sql, train_schema):
    target_sql = []
    for sql in total_sql:
        if -999 in [i[0] for i in sql['sql']['where']] or -1 in [i[0] for i in sql['sql']['where']]:
            continue
        if sql['db_name'] not in train_schema:
            continue
        else:
            target_sql.append(sql)
    return target_sql

def load_dataset(mode='train'):
    with open('data/train/train.json', 'r') as f:
        train_sql = json.loads(f.read())
    with open('data/train/db_schema.json', 'r') as f:
        d_schema = json.loads(f.read())
    if mode == 'test':
        with open('data/val/val_nosql.json', 'r') as f:
            dev_sql = json.loads(f.read())
    with open('data/val/val.json', 'r') as f:
        dev_sql = json.loads(f.read())
    with open('data/val/db_schema.json', 'r') as f:
        t_schema = json.loads(f.read())

    train_schema = {}
    dev_schema = {}
    for schema in t_schema:
        train_schema[schema['db_name']] = schema
    for schema in d_schema:
        dev_schema[schema['db_name']] = schema
    if mode == 'test':
        return dev_sql,dev_schema
    if mode == 'train':
        train_sql = filter(train_sql, train_schema)
        dev_sql = filter(dev_sql, dev_schema)
    return train_sql, train_schema, dev_sql, dev_schema

def to_batch_seq(sql_data, db_schema, idxes, st, ed, tokenizer=None, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    gt_cond_seq = []
    vis_seq = []
    sel_num_seq = []

    # need re-define
    lim_num_seq = []
    order_num_seq = []
    gt_having_cond_seq = []
    having_num_seq = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        sel_num = len(sql['sql']['select'])
        sel_num_seq.append(sel_num)

        db = db_schema[sql['db_name']]

        if len(sql['sql']['orderBy']) > 0:
            order_num_seq.append(1)
        else:
            order_num_seq.append(0)
        having_num_seq.append(math.ceil(len(sql['sql']['having']) / 2))
        lim_num = sql['sql']['limit'] if sql['sql']['limit'] else 0
        conds_num = math.ceil(len(sql['sql']['where']) / 2)

        cols = db['col_name']
        tablenames = db['table_name']

        # tokenizer = None
        if tokenizer:
            # print(sql['question'])
            q = tokenizer.tokenize(strPreProcess(sql['question']))
            tmp_col_seq = [["*"]]
            for col in cols[1:]:
                table_index = int(col[0].split('_')[-1])
                tablename = tablenames[table_index]
                tmp_col_seq.append(tokenizer.tokenize(strPreProcess(tablename + '.' + col[1])))

            # print("tokenizer")
            # print(q)
            # print(tmp_col_seq)

        else:
            q = [char for char in sql['question']]
            tmp_col_seq = [["*"]]
            for col in cols[1:]:
                table_index = int(col[0].split('_')[-1])
                tablename = tablenames[table_index]
                tmp_col_seq.append(list(tablename + '.' + col[1]))

            # print("no tokenizer")
            # print(q)
            # print(tmp_col_seq)

        q_seq.append(q)
        col_seq.append(tmp_col_seq)
        col_num.append(len(db_schema[sql['db_name']]['col_name']))

        try:
            if sql['sql']['where'][1] == 'AND':
                cond_conn_op = 1
            elif sql['sql']['where'][1] == 'OR':
                cond_conn_op = 2
        except:
            cond_conn_op = 0
        try:
            if sql['sql']['orderBy'][0] == 'DESC':
                order_sort = 1
            elif sql['sql']['orderBy'][0] == 'ASC':
                order_sort = 2
            order_col = [sql['sql']['orderBy'][1][0][0]]
            order_col_agg = [sql['sql']['orderBy'][1][0][1]]
        except:
            order_sort = 0
            order_col = []
            order_col_agg = []

        try:
            groupby_col = sql['sql']['groupby'][0]
        except:
            groupby_col = []

        try:
            if sql['sql']['having'][1] == 'AND':
                having_conn_op = 1
            elif sql['sql']['having'][1] == 'OR':
                having_conn_op = 2
        except:
            having_conn_op = 0

        ans_seq.append(
            (
                sel_num,
                [i[0] for i in sql['sql']['select']],
                [i[1] for i in sql['sql']['select']],
                conds_num,
                tuple(x[0] for x in sql['sql']['where'] if isinstance(x, list)),
                tuple(x[2] for x in sql['sql']['where'] if isinstance(x, list)),
                cond_conn_op,
                tuple(x[0][0] for x in sql['sql']['orderBy'] if isinstance(x, list)),
                order_sort,
                order_col,
                order_col_agg,
                len(groupby_col),
                groupby_col,
                lim_num,
                math.ceil(len(sql['sql']['having']) / 2),
                tuple(x[0] for x in sql['sql']['having'] if isinstance(x, list)),
                tuple(x[2] for x in sql['sql']['having'] if isinstance(x, list)),
                tuple(x[1] for x in sql['sql']['having'] if isinstance(x, list)),
                having_conn_op

            ))
        gt_cond_seq.append(
            [[x[0], x[2], str(x[3]).replace('"', '')] for x in sql['sql']['where'] if isinstance(x, list)])
        gt_having_cond_seq.append(
            [[x[0], x[2], str(x[3]).replace('"', '')] for x in sql['sql']['having'] if isinstance(x, list)])

        vis_seq.append((sql['question'], ''))
    if ret_vis_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, lim_num_seq, gt_having_cond_seq, having_num_seq, order_num_seq, vis_seq
    return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, lim_num_seq, gt_having_cond_seq, having_num_seq, order_num_seq

def pad_batch_seqs(seqs, pad=None, max_len=None):
    if not max_len:
        max_len = max([len(s) for s in seqs])
    if not pad:
        pad = 0
    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            seqs[i] = seqs[i][:max_len]
        else:
            seqs[i].extend([pad] * (max_len - len(seqs[i])))

    return seqs

def gen_batch_bert_seq(tokenizer, q_seq, col_seq, header_type=None, max_len=230):
    input_seq = []  # 输入编号
    q_mask = []  # NL mask
    sel_col_mask = []  # columns mask
    sel_col_index = []  # columns starting index
    where_col_mask = []
    where_col_index = []
    token_type_ids = []  # sentence A/B
    attention_mask = []  # length mask

    col_end_index = []

    q_lens = []
    sel_col_nums = []
    where_col_nums = []

    batch_size = len(q_seq)
    # print(batch_size)
    for i in range(batch_size):
        text_a = ['[CLS]'] + q_seq[i] + ['[SEP]']
        text_b = []
        for col_idx, col in enumerate(col_seq[i]):
            new_col = []
            if header_type is None:
                type_token1 = '[unused3]'
                type_token2 = '[unused6]'
                type_token3 = '[unused9]'
            elif header_type[i][col_idx] == 'text':
                type_token1 = '[unused1]'
                type_token2 = '[unused4]'
                type_token3 = '[unused7]'
            elif header_type[i][col_idx] == 'real':
                type_token1 = '[unused2]'
                type_token2 = '[unused5]'
                type_token3 = '[unused8]'

            new_col.extend(col)
            new_col.append(type_token2)  # type特征 用来分类第一次作为条件
            new_col.append(type_token3)  # type特征 用来分类第二次作为条件
            # TODO: 可以再加入新的标签来支持更多的列
            new_col.append(type_token1)  # type特征 用来分类sel, 同时分隔列名
            print(new_col)

            if len(text_a) + len(text_b) + len(new_col) >= max_len:
                break
            text_b.extend(new_col)

        text_b.append('[SEP]')

        inp_seq = text_a + text_b
        input_seq.append(inp_seq)
        q_mask.append([1] * (len(text_a) - 2))
        q_lens.append(len(text_a) - 2)
        token_type_ids.append([0] * len(text_a) + [1] * len(text_b))
        attention_mask.append([1] * len(inp_seq))

        sel_col = []
        where_col = []
        col_ends = []
        for i in range(len(text_a) - 1, len(inp_seq)):
            if inp_seq[i] in ['[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]']:
                where_col.append(i)
            if inp_seq[i] in ['[unused1]', '[unused2]', '[unused3]']:
                sel_col.append(i)
                col_ends.append(i)

        sel_col_mask.append([1] * len(sel_col))
        where_col_mask.append([1] * len(where_col))
        sel_col_nums.append(len(sel_col))
        where_col_nums.append(len(where_col))
        sel_col_index.append(sel_col)
        where_col_index.append(where_col)
        col_end_index.append(col_ends)

    #规范输入为同一长度，pad = ’[pad]‘ | 0
    input_seq = pad_batch_seqs(input_seq, '[PAD]')
    input_seq = [tokenizer.convert_tokens_to_ids(sq) for sq in input_seq] #字符token转化为词汇表里的编码id
    q_mask = pad_batch_seqs(q_mask)
    sel_col_mask = pad_batch_seqs(sel_col_mask)
    sel_col_index = pad_batch_seqs(sel_col_index)
    where_col_mask = pad_batch_seqs(where_col_mask)
    where_col_index = pad_batch_seqs(where_col_index)
    token_type_ids = pad_batch_seqs(token_type_ids)
    attention_mask = pad_batch_seqs(attention_mask)
    col_end_index = pad_batch_seqs(col_end_index)

    return (input_seq, q_mask, sel_col_mask, sel_col_index, where_col_mask, where_col_index, col_end_index,
            token_type_ids, attention_mask), q_lens, sel_col_nums, where_col_nums

def gen_bert_labels(q_seq, q_lens, sel_col_nums, where_col_nums, ans_seq, gt_cond_seq, header_type, col_seq):
    q_max_len = max(q_lens)
    sel_col_max_len = max(sel_col_nums)
    where_col_max_len = max(where_col_nums) #2col

    # labels init
    where_conn_label = np.array([x[6] for x in ans_seq])  # (None, )
    sel_num_label = np.array([0 for _ in ans_seq])  # (None, )
    where_num_label = np.array([0 for _ in ans_seq])  # (None, )
    sel_col_label = np.array([[0] * sel_col_max_len for _ in ans_seq], dtype=np.float)  # (None, col_max_len)
    sel_agg_label = np.array([[-1] * sel_col_max_len for _ in ans_seq])  # (None, col_max_len)
    where_col_label = np.array([[0] * where_col_max_len for _ in ans_seq], dtype=np.float)  # (None, 2col_max_len)
    where_op_label = np.array([[-1] * where_col_max_len for _ in ans_seq])  # (None, 2col_max_len)

    where_start_label = np.array([[-1] * where_col_max_len for _ in ans_seq])
    where_end_label = np.array([[-1] * where_col_max_len for _ in ans_seq])
    for b in range(len(gt_cond_seq)): # batch_size
        num_conds = len(gt_cond_seq[b]) # 条件数量
        if num_conds == 0:
            where_col_label[b] = 1.0 / sel_col_nums[b]  # 分散
            mass = 0
        else:
            mass = 1 / num_conds
        col_cond_count = {}
        for cond in gt_cond_seq[b]:
            if cond[0] >= sel_col_nums[b]:
                continue

            if cond[0] in col_cond_count:
                col_cond_count[cond[0]] += 1
            else:
                col_cond_count[cond[0]] = 0

            col_idx = 2 * cond[0] + col_cond_count[cond[0]] % 2
            where_op_label[b][col_idx] = cond[1]
            where_col_label[b][col_idx] += mass
            s, e = pos_in_tokens(cond[2], q_seq[b], header_type[b][cond[0]], col_seq[b][cond[0]])
            if s >= 0:
                s = min(s, q_lens[b] - 1)
                e = min(e - 1, q_lens[b] - 1)
                where_start_label[b][col_idx] = s
                where_end_label[b][col_idx] = e

        if num_conds > 0:
            where_num_label[b] = (where_col_label[b] > 0).sum()

        for b in range(len(ans_seq)):
            _sel = ans_seq[b][1]
            _agg = ans_seq[b][2]
            sel, agg = [], []
            for i in range(len(_sel)):
                if _sel[i] < sel_col_nums[b]:
                    sel.append(_sel[i])
                    agg.append(_agg[i])
            sel_num_label[b] = len(sel)
            mass = 1 / sel_num_label[b]
            if sel_num_label[b] == 0:
                mass = 1 / sel_col_nums[b]
            sel_col_label[b][sel] = mass
            sel_agg_label[b][sel] = agg

    return where_conn_label, sel_num_label, where_num_label, sel_col_label, sel_agg_label, \
           where_col_label, where_op_label, where_start_label, where_end_label

def to_batch_seq_test(sql_data, db_schema,idxes, st, ed):
    q_seq = []
    col_seq = []
    col_num = []
    raw_seq = []
    db_names = []
    question_ids = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        question_ids.append(sql['question_id'])
        q_seq.append([char for char in sql['question']])
        db = db_schema[sql['db_name']]
        tablenames = db['table_name']
        cols = db['col_name']
        tmp_col_seq = [["*"]]
        for col in cols[1:]:
            table_index = int(col[0].split('_')[-1])
            tablename = tablenames[table_index]
            tmp_col_seq.append(list(tablename + '.' + col[1]))
        col_seq.append(tmp_col_seq)
        col_num.append(len(db_schema[sql['db_name']]['col_name']))
        raw_seq.append(sql['question'])
        db_names.append(sql_data[idxes[i]]['db_name'])
    return q_seq, col_seq, col_num, raw_seq, db_names,question_ids

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    op_dict = {'AND': 1, 'OR': 2}
    for i in range(st, ed):
        sql = sql_data[idxes[i]]['sql']
        sql_new = {}
        sql_new['agg'] = [i[1] for i in sql["select"]]
        sql_new['sel'] = [i[0] for i in sql["select"]]
        sql_new['cond_conn_op'] = 0
        sql_new['having_conn_op'] = 0
        if len(sql["where"]) > 1:
            sql_new['cond_conn_op'] = op_dict[sql["where"][1]]
        if len(sql["having"]) > 1:
            sql_new['having_conn_op'] = op_dict[sql["having"][1]]
        sql_new['conds'] = [[max(0, i[0]), i[2], i[3]] for i in sql['where'] if isinstance(i, list)]
        sql_new['orderby_sort'] = 0
        sql_new['orderby'] = []
        sql_new['orderby_agg'] = []
        if len(sql['orderBy']) > 0:
            if sql['orderBy'][0] == 'DESC':
                sql_new['orderby_sort'] = 1
            else:
                sql_new['orderby_sort'] = 2
            sql_new['orderby'].append(sql['orderBy'][1][0][0])
            sql_new['orderby_agg'].append(sql['orderBy'][1][0][1])
        sql_new['groupby'] = sql['groupBy']
        sql_new['limit'] = sql['limit'] if sql['limit'] else 0
        sql_new['having'] = [[max(0, i[0]), i[2], i[3]] for i in sql['having'] if isinstance(i, list)]
        sql_new['having_agg'] = [[i[1]] for i in sql['having'] if isinstance(i, list)]
        sql_new['except'] = sql['except']
        sql_new['union'] =sql['union']
        sql_new['intersect'] =sql['intersect']
        query_gt.append(sql_new)
    return query_gt, ""


def epoch_train(model, optimizer, batch_size, sql_data, table_data, tokenizer=None):
    model.train()
    perm = range(len(sql_data))
    #perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    batch_size = 2
    for st in tqdm(range(math.ceil(len(sql_data) / batch_size))):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size

        if isinstance(model, SQLBert):
            # bert training
            q_seq, gt_sel_num, col_seq, col_num, ans_seq, \
            gt_cond_seq, gt_lim_num, gt_having_cond_seq, gt_having_num, gt_orderby_num \
                = to_batch_seq(sql_data, table_data, perm, st, ed, tokenizer)

            bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq)
            logits = model.forward(bert_inputs)  # condconn_logits, condop_logits, sel_agg_logits, q2col_logits
            #
            # # gen label
            labels = gen_bert_labels(q_seq, q_lens, sel_col_nums, where_col_nums, ans_seq, gt_cond_seq, header_type,
                                     col_seq)
            # q_seq  (12,q_lens) 问题内容
            # q_lens  (12,1)问题长度
            # sel_col_nums (12,1) col 长度
            # where_col_nums (12,1)2col长度
            # ans_seq   [(1, [6], [0], 1, (1,), (2,), 0),] len(agg),sel_col,agg,len(con),con_col,con_type,con_op
            # gt_cond_seq (12,3)条件列--列号，类型，值

            # compute loss
            # loss = model.loss(logits, labels, q_lens, sel_col_nums)
        else:
            print("no bert")

            q_seq, gt_sel_num, col_seq, col_num, ans_seq, \
            gt_cond_seq, gt_lim_num, gt_having_cond_seq, gt_having_num, gt_orderby_num \
                = to_batch_seq(sql_data, table_data, perm, st, ed)

            # q_seq: char-based sequence of question
            # gt_sel_num: number of selected columns and aggregation functions
            # col_seq: char-based column name
            # col_num: number of headers in one table
            # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
            # gt_cond_seq: ground truth of conds
            gt_where_seq = model.generate_gt_where_seq_test(q_seq, gt_cond_seq)
            gt_sel_seq = [x[1] for x in ans_seq]
            gt_orderby_seq = [x[9] for x in ans_seq]
            gt_having_seq = model.generate_gt_having_seq_test(q_seq, gt_having_cond_seq)
            gt_having_agg_seq = [x[15] for x in ans_seq]
            print(gt_sel_seq)
            print(gt_where_seq)
            print(gt_orderby_seq)
            print(gt_having_seq)
            print(gt_having_agg_seq)

            score = model.forward(q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq,
                                  gt_sel=gt_sel_seq,
                                  gt_sel_num=gt_sel_num, gt_orderby=gt_orderby_seq, gt_orderby_num=gt_orderby_num,
                                  gt_having=gt_having_seq, gt_having_agg=gt_having_agg_seq,
                                  gt_having_cond=gt_having_cond_seq)


        raise SystemExit

        # sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score

        # compute loss

        loss = model.loss(score, ans_seq, gt_where_seq, gt_having_seq)

        # gt_where_seq = model.generate_gt_where_seq_test(q_seq, gt_cond_seq)
        cum_loss += loss.data.cpu().numpy() * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cum_loss / len(sql_data)


def predict_test(model, batch_size, sql_data, table_data, output_path):
    model.eval()
    perm = list(range(len(sql_data)))
    fw = open(output_path, 'w')
    for st in tqdm(range(math.ceil(len(sql_data) / batch_size))):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        #q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, gt_lim_num, gt_having_cond_seq = to_batch_seq(
            #sql_data, table_data, perm,st, ed)
        q_seq, col_seq, col_num, raw_q_seq, table_ids, question_ids = to_batch_seq_test(sql_data, table_data, perm, st, ed)
        score = model.forward(q_seq, col_seq, col_num)
        _,sql_preds = model.gen_query(score, q_seq, col_seq, raw_q_seq)
        for index,sql_pred in enumerate(sql_preds):
            fw.write(str(question_ids[index])+'\t'+str(sql_pred)+'\n')
    fw.close()


def epoch_acc(model, batch_size, sql_data, table_data):
    model.eval()
    perm = list(range(len(sql_data)))
    badcase = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for st in tqdm(range(math.ceil(len(sql_data) / batch_size))):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, gt_lim_num, gt_having_seq, _,_,raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)

        # q_seq: char-based sequence of question
        # gt_sel_num: number of selected columns and aggregation functions, new added field
        # col_seq: char-based column name
        # col_num: number of headers in one table
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
        # gt_cond_seq: ground truth of conditions
        # raw_data: ori question, headers, sql
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        raw_q_seq = [x[0] for x in raw_data]  # original question
        ##
        try:
            score = model.forward(q_seq, col_seq, col_num)
        except:
            print(q_seq, col_seq, col_num)
            score = model.forward(q_seq, col_seq, col_num)
        pred_queries,_ = model.gen_query(score, q_seq, col_seq, raw_q_seq)
        # generate predicted format
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt)
        # except Exception as e:
        #     print(e)
        #     badcase += 1
        #     print 'badcase', badcase
        #     continue
        one_acc_num += (ed - st - one_err)
        tot_acc_num += (ed - st - tot_err)

    return one_acc_num / len(sql_data), tot_acc_num / len(sql_data)


def load_word_emb(file_name):
    print('Loading word embedding from %s' % file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
    return ret


thesaurus_dic = {
    '没有要求': '不限',
    '达标': '合格',
    '不': '否',
    '鄂': '湖北',
    '豫': '河南',
    '皖': '安徽',
    '冀': '河北',
    'inter': '因特尔',
    'samsung': '三星',
    '芒果TV': '湖南卫视',
    '湖南台': '芒果TV',
    '企鹅公司': '腾讯',
    '鹅厂': '腾讯',
    '宁': '南京',
    'Youku': '优酷',
    '荔枝台': '江苏卫视',
    '周一': '星期一',
    '周二': '星期二',
    '周三': '星期三',
    '周四': '星期四',
    '周五': '星期五',
    '周六': '星期六',
    '周日': '星期天',
    '周天': '星期天',
    '电视剧频道': '中央台八套',
    '沪': '上海',
    '闽': '福建',
    '增持': '买入'
}