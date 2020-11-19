import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
    parser.add_argument('--output_dir', type=str, default='./saved_model/res.txt',
                        help='Output path of prediction result')
    args = parser.parse_args()

    gpu = args.gpu
    n_word = 300
    batch_size = 30
    dev_sql, dev_schema = load_dataset(mode='test')

    word_emb = load_word_emb('data/char_embedding')
    model = SQLNet(word_emb, N_word=n_word, gpu=gpu)

    model_path = 'saved_model/best_model'
    print("Loading from %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    print("Loaded model from %s" % model_path)

    # dev_acc = epoch_acc(model, batch_size, dev_sql, dev_schema)
    # print ('Dev Logic Form Accuracy: %.3f' % (dev_acc[1]))

    print("Start to predict test set")
    predict_test(model, batch_size, dev_sql, dev_schema, args.output_dir)
    print("Output path of prediction result is %s" % args.output_dir)
