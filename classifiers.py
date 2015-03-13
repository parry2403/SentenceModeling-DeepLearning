import numpy as np
import scipy.sparse as sp
from collections import defaultdict, Counter
import sys, re, cPickle, random, logging, argparse

from sklearn import svm
from sklearn.metrics import accuracy_score

from process_data import WordVecs
from conv_net_sentence import get_cnn_representation

logger = logging.getLogger("customized.twsent.classifier")

def svm_train(train_xs, val_xs, test_xs, train_ys, val_ys, test_ys, nb=0, regs=[1.0]):
    """
    linear SVM training and test: report average performance over cross val folders
    """
    avg_val_perf, avg_corr_test_perf, avg_test_perf = 0,0,0
    for train_x,val_x,test_x,train_y,val_y,test_y in zip(train_xs,val_xs,test_xs,train_ys,val_ys,test_ys):
        
        if nb == 1: # NBSVM
            r = nbsvm(train_x, train_y, alpha=0.1)
            train_data, val_data, test_data = r[train_x.indices], r[val_x.indices], r[test_x.indices]
            train_x = sp.csr_matrix((train_data, train_x.indices, train_x.indptr), shape=train_x.shape)
            val_x = sp.csr_matrix((val_data, val_x.indices, val_x.indptr), shape=val_x.shape)
            test_x = sp.csr_matrix((test_data, test_x.indices, test_x.indptr), shape=test_x.shape)

        best_val_perf, corr_test_perf, best_test_perf = 0, 0, 0
        for reg in regs:
            clf = svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=reg, multi_class='ovr',
                     fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
            clf = clf.fit(train_x, train_y)
            ypred = clf.predict(val_x)
            val_perf = avg_fscore(ypred, val_y)
            ypred = clf.predict(test_x)
            test_perf = avg_fscore(ypred, test_y)
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                corr_test_perf = test_perf
                best_reg = reg
            if test_perf > best_test_perf:
                best_test_perf = test_perf
        avg_val_perf += best_val_perf
        avg_corr_test_perf += corr_test_perf
        avg_test_perf += best_test_perf
        #print "reg = %.2f, val F1 = %.4f, test F1 = %.4f, best test F1 = %.4f" %(best_reg, best_val_perf, corr_test_perf, best_test_perf)
    cv = len(train_xs)
    avg_val_perf /= cv
    avg_corr_test_perf /= cv
    avg_test_perf /= cv
    print "average val F1 = %.4f, average test F1 = %.4f, average best test F1 = %.4f" %(avg_val_perf, avg_corr_test_perf, avg_test_perf)
    return avg_val_perf, avg_corr_test_perf, avg_test_perf

def avg_fscore(y_pred, y_gold):
    pos_p, pos_g = sum(y_pred), sum(y_gold)
    neg_p, neg_g = len(y_pred)-pos_p, len(y_gold)-pos_g
    if pos_p==0 or pos_g==0 or neg_p==0 or neg_g==0: return 0.0
    pos_m, neg_m = 0, 0
    for p,g in zip(y_pred, y_gold):
        if p==g:
            if p == 1: pos_m += 1
            elif p == 0: neg_m += 1
    pos_prec, pos_reca = float(pos_m) / pos_p, float(pos_m) / pos_g
    neg_prec, neg_reca = float(neg_m) / neg_p, float(neg_m) / neg_g
    if pos_m == 0 or neg_m == 0: return 0.0
    pos_f1, neg_f1 = 2*pos_prec*pos_reca / (pos_prec+pos_reca), 2*neg_prec*neg_reca / (neg_prec+neg_reca)
    return (pos_f1+neg_f1)/2.0
#    return accuracy_score(y_pred, y_gold)

def make_data(dataset, grams=[1], use_cluster=0, clu_weight=1.0, use_pv=0, use_cnn=0, use_feature=0, cv=0):
    revs, wordvecs, max_l = dataset
    feats, yy = [], []
    for i,rev in enumerate(revs):
        sent = rev["text"]
        tokens = tokenize(sent, grams)
        if use_pv == 0:
            feat = dict(zip(tokens, np.ones(len(tokens)).tolist()))
            if use_cluster == 1:  # cluster specific features
                cluster = rev["cluster"]
                clu_toks = [tok+"_*_"+str(cluster) for tok in tokens]
                feat.update(dict(zip(clu_toks, (clu_weight * np.ones(len(clu_toks))).tolist())))
            elif use_cluster == 2:  # cluster as a single feature
                cluster = rev["cluster"]
                feat.update({"user_clusterID="+str(cluster):1})
        else:
            feat = rev["pv"]

        if use_feature == 1:  # hand-crafting features
            feat.update(rev["features"])

        feats.append(feat)
        yy.append(rev["y"])
    yy = np.array(yy,dtype="int")
    xx = process_sparse_feats(feats)

    train_xs, val_xs, test_xs, train_ys, val_ys, test_ys = [], [], [], [], [], []
    if cv == 0:
        val_fold, test_fold = 2, 3
        train, val, test = [], [], []
        for i,rev in enumerate(revs):
            if rev["split"]==val_fold:
                val.append(i)
            elif rev["split"]==test_fold:  
                test.append(i)
            else:
                train.append(i)
        train = np.array(train,dtype="int")
        val = np.array(val,dtype="int")
        test = np.array(test,dtype="int")
        train_ys.append(yy[train])
        val_ys.append(yy[val])
        test_ys.append(yy[test])
        if use_cnn == 0 or wordvecs is None:
            train_xs.append(xx[train])
            val_xs.append(xx[val])
            test_xs.append(xx[test])
        else:
            tmp = get_cnn_representation(revs, wordvecs.W, wordvecs.word_idx_map, max_l, val_test_splits=[val_fold, test_fold])
            train_xs.append(tmp[0])
            val_xs.append(tmp[1])
            test_xs.append(tmp[2])
    else:
        for fold in xrange(cv):
            test_fold, val_fold = fold, (fold + 1) % cv
            train, val, test = [], [], []
            for i,rev in enumerate(revs):
                if rev["split"]==test_fold:  
                    test.append(i)
                elif rev["split"]==val_fold:
                    val.append(i)
                else:
                    train.append(i)
            train = np.array(train,dtype="int")
            val = np.array(val,dtype="int")
            test = np.array(test,dtype="int")
            train_ys.append(yy[train])
            val_ys.append(yy[val])
            test_ys.append(yy[test])
            if use_cnn == 0 or wordvecs is None:
                train_xs.append(xx[train])
                val_xs.append(xx[val])
                test_xs.append(xx[test])
            else:
                tmp = get_cnn_representation(revs, wordvecs.W, wordvecs.word_idx_map, max_l, val_test_splits=[val_fold, test_fold])
                train_xs.append(tmp[0])
                val_xs.append(tmp[1])
                test_xs.append(tmp[2])

    return [train_xs, val_xs, test_xs, train_ys, val_ys, test_ys]


def nbsvm(train_x, train_y, alpha=1):
    """
    Naive Bayes SVM (Want and Manning, 2012)
    take csr_matrix and label array as input directly
    """
    pos_idx, neg_idx = np.where(train_y==1)[0], np.where(train_y==0)[0]
    d = xx.shape[1]
    p = np.asarray(train_x[pos_idx].sum(0)).flatten() + np.ones(d) * alpha
    q = np.asarray(train_x[neg_idx].sum(0)).flatten() + np.ones(d) * alpha
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return r

def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens

def process_sparse_feats(sp_feats):
    feat_idx = {}
    row, col, data = [], [], []
    for i,feat_map in enumerate(sp_feats):
        for feat in feat_map.keys():
            if feat not in feat_idx:
                feat_idx[feat] = len(feat_idx)
            row.append(i)
            col.append(feat_idx[feat])
            data.append(feat_map[feat])
    xx = sp.csr_matrix((data,(row,col)), shape=(len(sp_feats), len(feat_idx)))
    return xx


if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="name of dataset: omd or semeval")
    parser.add_argument("--use_cluster", type=int, default=0, help="use user cluster information for feature conjunction (1) or as an additional feature (2) or not (0), default is 0")
    parser.add_argument("--use_pv", type=int, default=0, help="use paragraph vector (1) or not (0), default is 0")
    parser.add_argument("--use_cnn", type=int, default=0, help="use CNN (1) or not (0), default is 0")
    parser.add_argument("--use_feature", type=int, default=0, help="use hand-crafting features (1) or not (0), default is 0")
    parser.add_argument("--n_gram", type=int, default=1, help="use n-gram features up to n_gram, default is 1 (only include unigrams)")
    parser.add_argument("--nb", type=int, default=0, help="use NBSVM (1) or not (0), default is 0")
    args = parser.parse_args()

    if args.dataset_name == "omd":
        fname = "omd.pkl"
        cv = 10
    elif args.dataset_name == "semeval":
        fname = "semeval.pkl"
        cv = 0
    else:
        print "only omd or semeval is supported for now"
        sys.exit()

    dataset = cPickle.load(open(fname,"rb"))
    grams = range(1, args.n_gram+1)

    best_val_perf, best_corr_test_perf, best_test_perf = 0, 0, 0
    if args.use_cluster == 0 or args.use_cluster == 2:
        weights = [1.0]
    else:
        weights = [0.1, 0.25, 0.5, 0.75, 1.0] # weight for cluster features
    for weight in weights:
        logger.info("start to build train/val/test datasets, n-gram up to %d, user_cluster=%d, clu_weight=%.2f, use_feature=%d" %(args.n_gram, args.use_cluster, weight, args.use_feature))
        train_xs, val_xs, test_xs, train_ys, val_ys, test_ys = make_data(dataset, grams=grams, use_cluster=args.use_cluster, clu_weight=weight, use_pv=args.use_pv, use_cnn=args.use_cnn, use_feature=args.use_feature, cv=cv)
        regs = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 4.0]
        val_perf, corr_test_perf, test_perf = svm_train(train_xs, val_xs, test_xs, train_ys, val_ys, test_ys, nb=args.nb, regs=regs)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            best_corr_test_perf = corr_test_perf
        if test_perf > best_test_perf:
            best_test_perf = test_perf
    print "Final results: val_perf = %.4f, test_perf = %.4f, best_test_perf = %.4f" %(best_val_perf, best_corr_test_perf, best_test_perf)

    logger.info('end logging')
