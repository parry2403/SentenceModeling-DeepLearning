import numpy as np
import scipy.sparse as sp
import cPickle
from collections import defaultdict
import sys, re, os, logging, argparse
import pandas as pd

from sklearn import cluster

import gensim
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from gensim import corpora, models, similarities
import feature

logger = logging.getLogger("customized.twsent.procdata")

def build_user_graph_CNN(fname,user_idx_map):
    user_edges = defaultdict(list)
    with open(fname, "rb") as f:
        for line in f:
            edge = line.strip().split()
            user = edge[0].strip()
            follower = edge[1].strip()

            if user in user_idx_map.keys() and follower in user_idx_map.keys():
                  user_edges[user_idx_map[user]].append(user_idx_map[follower])
    return user_edges

def build_user_data_CNN(fname):
    cv=10
    user_vocab = defaultdict(float)
    with open(fname, "rb") as f:
        for line in f:
            line = line.strip()
            spidx = line.rfind("|")

            rev = line[spidx:].strip().split(",")[1]
            user_vocab[rev] += 1


    return user_vocab
def build_data_CNN(fname,user_idx_map):
    cv=10
    vocab = defaultdict(float)
    with open(fname, "rb") as f:
        for line in f:
            line = line.strip()
            spidx = line.rfind("|")
            metas = line[spidx+1:].strip().split(",")
            label = int(metas[0])
            rev = line[:spidx].strip()
            orig_rev = clean_str(rev)
            # logger.info(orig_rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":label,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv),
                      "user":user_idx_map[metas[1]]}
            revs.append(datum)
    max_l = np.max(pd.DataFrame(revs)["num_words"])

    return revs,vocab,max_l
def build_data(fname, user_cluster=None, pv_model=None, cv=10):
    """
    Loads and process data.
    """
    feature_extractor = feature.Feature()
    
    revs = []
    vocab = defaultdict(float)
    ins_idx, tw_w_clu = 0, 0
    users = set()

    if pv_model is not None:
        dim = pv_model.layer1_size
        keys = ["PV_%d" %i for i in range(dim)]
        if user_cluster is not None: key_clus = ["PV_clu_%d" %i for i in range(dim)]

    with open(fname, "rb") as f:
        for line in f:  
            line = line.strip()
            spidx = line.rfind("|")
            rev = line[:spidx].strip()
            metas = line[spidx+1:].strip().split(",")
            label = int(metas[0])
            cluster = 0
            if user_cluster is not None and metas[1] in user_cluster: 
                cluster = user_cluster[metas[1]]
                users.add(metas[1])
                tw_w_clu += 1
            if len(metas) > 2:
                split = int(metas[2])
            else:
                #split = np.random.randint(0,cv)
                split = ins_idx % cv
            nrc_feat = feature_extractor.NRC_feature_extractor(line)
            orig_rev = clean_str(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            if pv_model is not None:
                pv_dict = dict(zip(keys, pv_model['SENT_%d' % ins_idx]))
                if user_cluster is not None:
                    pv_dict.update(dict(zip(key_clus, pv_model['CLUSTER_%d' % cluster])))
            else:
                pv_dict = {}
            datum  = {"y":label, 
                      "text": orig_rev,                             
                      "features": nrc_feat,
                      "pv": pv_dict,
                      "num_words": len(orig_rev.split()),
                      "cluster":cluster,
                      "split": split}
            revs.append(datum)
            ins_idx += 1
    max_l = np.max(pd.DataFrame(revs)["num_words"])

    logger.info("finish building data: %d tweets, in which %d tweets of %d users have link information" %(ins_idx, tw_w_clu, len(users)))
    logger.info("tweets of users have no links are associated with cluster ID 0 (the biggest cluster)")
    logger.info("vocab size: %d, max tweet length: %d" %(len(vocab), max_l))
    return revs, vocab, max_l
    

def graph_cluster(fname, clu_algo="spectral", n_clusters=5):
    user_idx = {}
    row, col, data = [], [], []
    if clu_algo == "spectral" or clu_algo == "ap":  # define affinity matrix directly
        with open(fname, "rb") as f:
            for line in f:
                uids = line.strip().split()
                if uids[0] not in user_idx: user_idx[uids[0]] = len(user_idx)
                if uids[1] not in user_idx: user_idx[uids[1]] = len(user_idx)
                row.append(user_idx[uids[0]])
                col.append(user_idx[uids[1]])
                data.append(1)
                row.append(user_idx[uids[1]])
                col.append(user_idx[uids[0]])
                data.append(1)
        xx = sp.csr_matrix((data,(row,col)), shape=(len(user_idx), len(user_idx)))
    else:
        feat_idx = {}
        with open(fname, "rb") as f:
            for line in f:
                uids = line.strip().split()
                if uids[0] not in user_idx: user_idx[uids[0]] = len(user_idx)
                if uids[1] not in feat_idx: feat_idx[uids[1]] = len(feat_idx)
                row.append(user_idx[uids[0]])
                col.append(feat_idx[uids[1]])
                data.append(1)
        xx = sp.csr_matrix((data,(row,col)), shape=(len(user_idx), len(feat_idx)))

    if clu_algo == "spectral":
        clu = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        clu.fit(xx)
    elif clu_algo == "ap":
        clu = cluster.AffinityPropagation(affinity='precomputed')
        clu.fit(xx.todense())
    elif clu_algo == "kmeans":
        clu = cluster.KMeans(n_clusters=n_clusters, n_jobs=20)
        clu.fit(xx)
    elif clu_algo == "agglo":
        clu = cluster.AgglomerativeClustering(n_clusters=n_clusters)
        clu.fit(xx)
    user_cluster = {uid:clu.labels_[user_idx[uid]] for uid in user_idx}
    logger.info("finish clustering users and the total number of clusters is %d" %(max(clu.labels_)+1))
    return user_cluster

def para_vector(fname, unlabel_fname=None, clu_algo=None, epoch=100, dim=100, negative=15):
    sentences = []
    ins_idx = 0
    with open(fname, "rb") as f:
        for line in f:  
            line = line.strip()
            spidx = line.rfind("|")
            rev = line[:spidx].strip()
            if clu_algo is not None:
                metas = line[spidx+1:].strip().split(",")
                cluster = 0
                if metas[1] in user_cluster: 
                    cluster = user_cluster[metas[1]]
            orig_rev = clean_str(rev)
            for epo in xrange(epoch):
                if clu_algo is not None:
                    sentences.append(LabeledSentence(words=orig_rev.split(), labels=['SENT_%d' % ins_idx, 'CLUSTER_%d' % cluster]))
                else:
                    sentences.append(LabeledSentence(words=orig_rev.split(), labels=['SENT_%d' % ins_idx]))
            ins_idx += 1

    if unlabel_fname is not None:
        with open(unlabel_fname, "rb") as f:
            for line in f:  
                rev = line.strip()
                orig_rev = clean_str(rev)
                sentences.append(LabeledSentence(words=orig_rev.split(), labels=['SENT_%d' % ins_idx]))
                ins_idx += 1

    rand_idx = np.random.permutation(len(sentences))
    sentences = [sentences[idx] for idx in rand_idx]
    model = Doc2Vec(sentences, size=dim, min_count=1, workers=20, dm=0, hs=0, negative=negative)
    return model

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()
class UserWordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab):

        word_vecs = self.load_bin_vec(fname, vocab)

        self.k = len(word_vecs.values()[0])
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                   # logger.info(word_vecs[word])
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def load_txt_vec(self, fname, vocab, has_header=False):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            if has_header: header = f.readline()
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)


class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1, has_header=False):
        if binary == 1:
            word_vecs = self.load_bin_vec(fname, vocab)
        else:
            word_vecs = self.load_txt_vec(fname, vocab, has_header)
        self.k = len(word_vecs.values()[0])
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))            
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                   # logger.info(word_vecs[word])
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs
    
    def load_txt_vec(self, fname, vocab, has_header=False):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            if has_header: header = f.readline()
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
    

if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset_name", help="name of dataset: omd or semeval")
    # parser.add_argument("--graph", default="fnet", help="social graph: follower network (full or not) or mention network (full or not). Selected from {fnet, fnetf, mnet, mnetf}, default is fnet")
    # parser.add_argument("--clu_algo", help="clustering algorithm: Kmeans, Spectral Clustering, Affinity Propagation, Agglomerative Clustering. Selected from {kmeans, spectral, ap, agglo}")
    # parser.add_argument("--n_clu", type=int, default=5, help="number of clusters, only works for spectral clustering setting, default is 5")
    # parser.add_argument("--use_pv", type=int, default=0, help="use paragraph vector or not, default is 0")
    # parser.add_argument("--w2v_fname", help="path/name of pretrained word embeddings file")
    # args = parser.parse_args()
    #
    # if args.dataset_name == "omd":
    #     fname = "../data/twitter-sentiment/OMD/omd.txt"
    #     outfname = "omd.pkl"
    #     if args.graph == "fnet":
    #         netfname = "../data/twitter-sentiment/OMD/omd.net"
    #     elif args.graph == "fnetf":
    #         netfname = "../data/twitter-sentiment/OMD/omd-friends.net"
    #     elif args.graph == "mnet":
    #         netfname = "../data/twitter-sentiment/OMD/omd-mention.net"
    #     elif args.graph == "mnetf":
    #         netfname = "../data/twitter-sentiment/OMD/omd-full-mention.net"
    # elif args.dataset_name == "semeval":
    #     fname = "../data/twitter-sentiment/SemEval/semeval.txt"
    #     outfname = "semeval.pkl"
    #     if args.graph == "fnet":
    #         netfname = "../data/twitter-sentiment/SemEval/semeval.net"
    #     elif args.graph == "fnetf":
    #         netfname = "../data/twitter-sentiment/SemEval/semeval-friends.net"
    #     elif args.graph == "mnet":
    #         netfname = "../data/twitter-sentiment/SemEval/semeval-mention.net"
    #     elif args.graph == "mnetf":
    #         netfname = "../data/twitter-sentiment/SemEval/semeval-full-mention.net"
    # else:
    #     print "only omd or semeval is supported for now"
    #     sys.exit()
    #
    # user_cluster = None
    # if args.clu_algo is not None:
    #     logger.info("start clustering Twitter users by %s" %args.clu_algo)
    #     user_cluster = graph_cluster(netfname, clu_algo=args.clu_algo, n_clusters=args.n_clu)
    #
    # pv_model = None
    # unlabel_fname = "/nethome/corpora/edi-stuff/edinburgh_10M.en"
    # if args.use_pv == 1:
    #     logger.info("start learning paragraph vectors")
    #     pv_model = para_vector(fname, unlabel_fname=None, clu_algo=args.clu_algo, epoch=100, dim=100, negative=15)
    #
    # revs, vocab, max_l = build_data(fname, user_cluster=user_cluster, pv_model=pv_model)
    fname = "../data/twitter-sentiment/SemEval/semeval.txt"
    w2v_fname= "../data/word2vec/GoogleNews-vectors-negative300.bin"
    outfname = "semeval.pkl"
    # revs, vocab, max_l = build_data(fname)
    network = "../data/twitter-sentiment/SemEval/semeval.net"
   # Parry

    revs=[]


    user_vocab = build_user_data_CNN(fname)

    dictionary = corpora.Dictionary( [user_vocab.keys()])

    dictionary.save('../data/User_Dict.dict') # store the dictionary, for future reference

    k=300

    user_idx_map = dict()
    vocab_size = len(dictionary.keys())
    UserW=np.zeros(shape=(vocab_size, k))

    for i in range(0,vocab_size):
        UserW[i] = np.random.uniform(-0.25,0.25,k)
        user_idx_map[dictionary[i]]=i


    user_graph = build_user_graph_CNN(network,user_idx_map)
    print user_graph[1]
    revs, vocab, max_l= build_data_CNN(fname,user_idx_map)
    wordvecs = None
    W = None
    word_idx_map =None

    # if w2v_fname is not None: # use word embeddings for CNN
    logger.info("loading and processing pretrained word vectors")
    wordvecs = WordVecs(w2v_fname, vocab, binary=1, has_header=False)

    #cPickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open("semeval13_tweet.p", "wb"))
    cPickle.dump([revs, wordvecs.W, wordvecs.word_idx_map, max_l,UserW,user_idx_map,user_graph], open(outfname, "wb"))
    logger.info("dataset created!")
    logger.info("end logging")
    
