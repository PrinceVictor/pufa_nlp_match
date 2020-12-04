import json
import numpy as np
from tqdm import tqdm
import math



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

def to_batch_seq(sql_data, db_schema, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    gt_cond_seq = []
    vis_seq = []
    sel_num_seq = []
    lim_num_seq = []
    order_num_seq = []
    gt_having_cond_seq = []
    having_num_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        print(sql)
        db = db_schema[sql['db_name']]
        sel_num = len(sql['sql']['select'])
        sel_num_seq.append(sel_num)
        if len(sql['sql']['orderBy']) > 0:
            order_num_seq.append(1)
        else:
            order_num_seq.append(0)
        having_num_seq.append(math.ceil(len(sql['sql']['having']) / 2))
        lim_num = sql['sql']['limit'] if sql['sql']['limit'] else 0
        conds_num = math.ceil(len(sql['sql']['where']) / 2)
        q_seq.append([char for char in sql['question']])
        cols = db['col_name']
        tablenames = db['table_name']
        tmp_col_seq = [["*"]]
        for col in cols[1:]:
            table_index = int(col[0].split('_')[-1])
            tablename = tablenames[table_index]
            tmp_col_seq.append(list(tablename + '.' + col[1]))
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


def epoch_train(model, optimizer, batch_size, sql_data, table_data):
    model.train()
    perm = range(len(sql_data))
    #perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    batch_size = 1
    for st in tqdm(range(math.ceil(len(sql_data) / batch_size))):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, gt_lim_num, gt_having_cond_seq, gt_having_num, gt_orderby_num = to_batch_seq(
            sql_data, table_data, perm,
            st, ed)

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

        score = model.forward(q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq,
                              gt_sel_num=gt_sel_num, gt_orderby=gt_orderby_seq,gt_orderby_num=gt_orderby_num ,gt_having=gt_having_seq, gt_having_agg=gt_having_agg_seq,gt_having_cond=gt_having_cond_seq)
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
