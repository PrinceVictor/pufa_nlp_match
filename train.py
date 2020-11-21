import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number')
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu to train')
    parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.bs
    n_word = 300
    learning_rate = 3e-3

    # load dataset
    train_sql, train_schema, dev_sql, dev_schema = load_dataset()
    word_emb = load_word_emb('data/char_embedding')
    model = SQLNet(word_emb, N_word=n_word, gpu=gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    if args.restore:
        model_path = 'saved_model/test_model'
        print("Loading trained model from %s" % model_path)
        model.load_state_dict(torch.load(model_path))

    # used to record best score of each sub-task
    best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
    best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
    best_lf, best_lf_idx = -1.0, 0

    print("#" * 20 + "  Star to Train  " + "#" * 20)
    for i in range(args.epoch):
        print('Epoch %d' % (i + 1))
        # train on the train dataset
        train_loss = epoch_train(model, optimizer, batch_size, train_sql, train_schema)
        # evaluate on the dev dataset
        dev_acc = epoch_acc(model, batch_size, dev_sql, dev_schema)
        # accuracy of each sub-task
        print(
            'Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f, Ord-Col: %.3f, '
            'Ord-Sort: %.3f, Ord-Agg: %.3f, Grp-Exist: %.3f, Grp-Col: %.3f, Lim-Num: %.3f,Having-Col: %.3f, Having-Op: %.3f, Having-Val: %.3f,'
            ' Having-Agg: %.3f, Having-Num: %.3f, Having-Rel: %.3f, Except-Val: %.3f,Union-Val: %.3f, Intersect-Val: %.3f' % (
                dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6],
                dev_acc[0][7], dev_acc[0][8], dev_acc[0][9], dev_acc[0][10], dev_acc[0][11], dev_acc[0][12],
                dev_acc[0][13], dev_acc[0][14], dev_acc[0][15], dev_acc[0][16], dev_acc[0][17], dev_acc[0][18],
                dev_acc[0][19], dev_acc[0][20], dev_acc[0][21], dev_acc[0][22]))
        # save the best model
        if dev_acc[1] > best_lf:
            best_lf = dev_acc[1]
            best_lf_idx = i + 1
            torch.save(model.state_dict(), 'saved_model/test_model')
        print('Train loss = %.3f' % train_loss)
