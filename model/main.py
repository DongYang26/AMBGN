from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import SFGCN
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
from estimateA import EstimateAdj


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, required=True)
    parse.add_argument("-t", "--true", help="input int", type=int, required=True)
    parse.add_argument("-k", "--knn", help="Parameters of KNN", type=int, required=True)
    parse.add_argument("-Q", "--prob", help="Probability of edge connection", type=float, required=True)
    parse.add_argument("-s", "--st", help="stop EM algorithm", type=float, default=.01)

    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    print(torch.cuda.device_count())

    cuda = not config.no_cuda and torch.cuda.is_available()
    print(cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('test2')

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)
    if args.dataset != "pubmed" and args.dataset != "arxiv":
        sadj, fadjj, sadjh, fadjh = load_graph(args, config)
        data = load_data_my(config, sadj, sadjh, fadjh)
        features = data.x
        labels = data.y
        idx_train = data.idx_train
        idx_test = data.idx_test
        fadj = fadjj
    else:
        adj_gcn, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels, fadj, fadjh = data_partition_random(
            config, dataset_dir='../data/'+args.dataset+'/', dataset_name=args.dataset, label_n_per_class=args.labelrate)
        data = load_data__my(adj_gcn, node_features, one_hot_labels, train_mask, val_mask, test_mask, fadjh)
        sadj = data.adj
        features = data.x
        labels = data.y
        idx_train = data.idx_train
        idx_test = data.idx_test

    acc_max = 0
    f1_max = 0
    epoch_max = 0
    outputZ = None
    hidden_outputZ = None
    hidden_outputZ1 = None
    iter_max = 0
    acc_secondMax = 0
    f1_secondMax = 0
    att_max = None


    model = SFGCN(nfeat = config.fdim,
              nhid1 = config.nhid1,
              nhid2 = config.nhid2,
              nclass = config.class_num,
              n = config.n,
              dropout = config.dropout)
    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def train(model, epochs, trfadj):
        model.train()
        optimizer.zero_grad()
        output, att, emb1, com1, com2, emb2, emb, hidden_output = model(features, sadj, trfadj)
        # print(att)

        loss_class = F.nll_loss(output[idx_train], labels[idx_train])
        loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
        loss_com = common_loss(com1, com2)
        loss = loss_class + config.beta * loss_dep + config.theta * loss_com
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model, trfadj)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        data_preserve_step(att)
        return loss.item(), acc_test.item(), macro_f1.item(), emb2, output, hidden_output, att

    def main_test(model, tefadj):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb, hidden_outpute1 = model(features, sadj, tefadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb

    def train_model(acc_max, iter, tmfadj):
        global acc_secondMax, f1_secondMax, outputZ, hidden_outputZ, epoch_max, iter_max, f1_max, att_max, hidden_outputZ1
        for epoch in range(config.epochs):
            loss, acc_test, macro_f1, emb, output1, hidden_output1, att_max = train(model, epoch, tmfadj)

            if acc_test >= acc_max:
                acc_max = acc_test
                f1_max = macro_f1
                epoch_max = epoch
                outputZ = output1
                hidden_outputZ = hidden_output1
                iter_max = iter
                hidden_outputZ1 = emb

            if acc_secondMax <= acc_test < acc_max and iter != 0:
                acc_secondMax = acc_test
                f1_secondMax = macro_f1

        print('epoch:{}'.format(epoch_max),
              'iter:{}'.format(iter_max),
              'acc_max: {:.4f}'.format(acc_max),
              'f1_max: {:.4f}'.format(f1_max),
              'acc_secondMax: {:.4f}'.format(acc_secondMax),
              'f1_secondMax: {:.4f}'.format(f1_secondMax))
        return acc_max
    

    estimator = EstimateAdj(data, args)
    print('homophilys:{}'.format(data.homophilys))
    print('homophilyh:{}'.format(data.homophilyf))
    # best_homophily = data.homophilys
    best_homophily = 0
    best_featureG = sadj
    for iter in range(25):
        print('=======================Started at:{}th time======================='.format(iter))
        if iter:
            estimator.reset_obs()
            estimator.update_obs(knn(data, data.x, args.knn))
            estimator.update_obs(knn(data, hidden_outputZ, args.knn))
            estimator.update_obs(knn(data, hidden_outputZ1, args.knn))
            estimator.update_obs(knn(data, outputZ, args.knn))
            alpha, beta, O, Q, iterations = estimator.EM(outputZ.max(1)[1].detach().cpu().numpy(),
                                                         args.st)
            RA, RAhomophily = prob_to_adj(Q, args.prob, data)
            if RAhomophily > best_homophily:
                RAadj = F.normalize(RA, dim=0, p=1)
                best_homophily = RAhomophily
                best_featureG = RAadj.cuda()

            acc_max = train_model(acc_max, iter, best_featureG)
        if not iter:
            acc_max = train_model(acc_max, iter, fadj)
            # plt_show(outputZ.detach().cpu().numpy(), data, args, str(iter))
        data_preserve(att_max, iter, args)

    print('acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max),
          'best_homophily: {:.4f}'.format(best_homophily))

    # plt_show(outputZ.detach().cpu().numpy(), data, args)


