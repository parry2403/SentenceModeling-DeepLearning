"""
Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys

from sklearn import svm
warnings.filterwarnings("ignore")   

from conv_net_classes import *

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,
                   U, UserW,
                   userDictionary,user_graph,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))  # max pooling, only one
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')  # T.vector dtype=int32
    u = T.ivector('u')


    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))]) # updates: Words = T.set...
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch

    Users = theano.shared(value = UserW, name = "Users")

    # zero_vec_tensor = T.vector()
    # zero_vec = np.zeros(img_w)
    # set_zero = theano.function([zero_vec_tensor], updates=[(Users, T.set_subtensor(Users[0,:], zero_vec_tensor))]) # updates: Words = T.set...
    # UsInput = Users[T.cast(u.flatten(),dtype="int32")].reshape((x.shape[0],Users.shape[1]))     # input: word embeddings of the mini batch

    conv_layers = []        # layer number = filter number
    layer1_inputs = []      # layer number = filter number
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1) # concatenate representations of different filters
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    user_info = UserSpace(input=layer1_input)
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
      #  params += [Users]

    cost = classifier.negative_log_likelihood(y)
    #+user_info.user_cost(Users[T.cast(u.flatten(),dtype="int32")],Users)+user_info.user_graph_cost(Users[T.cast(u.flatten(),dtype="int32")],user_graph,Users))
    print  "test******"
    print cost
    dropout_cost =( classifier.dropout_negative_log_likelihood(y))
                    #+user_info.user_cost(Users[T.cast(u.flatten(),dtype="int32")],Users)+user_info.user_graph_cost(Users[T.cast(u.flatten(),dtype="int32")],user_graph,Users))
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    # usercost = user_info.user_cost(u)
    # g_W = T.grad(cost=usercost, wrt=classifier.W)
    # usergrad_updates
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = datasets[0]
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)

        user_set = datasets[3]
        extra_data = user_set[:extra_data_num]
        user_data=np.append(datasets[3],extra_data,axis=0)

    else:
        new_data = datasets[0]
        user_data = datasets[3]

    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))




    train_set = new_data
    val_set = datasets[1]
    user_data =  theano.shared( np.asarray(user_data ,  dtype=theano.config.floatX),borrow=True)

    user_data= T.cast(user_data, 'int32')
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    userval_set =datasets[4]
    userval_set =  theano.shared(np.asarray(userval_set,
                                                dtype=theano.config.floatX),
                                  borrow=True)



    train_set_x_org = datasets[0][:,:img_h]
    train_set_y_org = np.asarray(datasets[0][:,-1],"int32")
    val_set_x_org = datasets[1][:,:img_h] 
    val_set_y_org = np.asarray(datasets[1][:,-1],"int32")
    test_set_x = datasets[2][:,:img_h] 
    test_set_y = np.asarray(datasets[2][:,-1],"int32")
    usertest_set =  np.asarray(datasets[5],"int32")

    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size],
         #   u :user_data[index*batch_size:(index+1)*batch_size]
          })
    # train_model_user = theano.function([index], usercost, updates=usergrad_updates,
    #       givens={
    #         x: train_set_x[index*batch_size:(index+1)*batch_size],
    #        # y: train_set_y[index*batch_size:(index+1)*batch_size],
    #         u :user_data[index*batch_size:(index+1)*batch_size]
    #       })
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)

    test_model = theano.function([x], [test_layer1_input, test_y_pred])   
   
    #obtain CNN representations
    train_pred_layers = []
    train_size = train_set_x_org.shape[0]
    train_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((train_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        train_layer0_output = conv_layer.predict(train_layer0_input, train_size)
        train_pred_layers.append(train_layer0_output.flatten(2))
    train_layer1_input = T.concatenate(train_pred_layers, 1)
    train_pred_model = theano.function([x], train_layer1_input)

    val_pred_layers = []
    val_size = val_set_x_org.shape[0]
    val_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((val_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        val_layer0_output = conv_layer.predict(val_layer0_input, val_size)
        val_pred_layers.append(val_layer0_output.flatten(2))
    val_layer1_input = T.concatenate(val_pred_layers, 1)
    val_y_pred = classifier.predict(val_layer1_input)
    val_model = theano.function([x], [val_layer1_input, val_y_pred])   


    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    newxx = []
    mys = "Output-omd"  + ".txt"
    result = open(mys,"w")

#mys1 = "dep2"  + ".txt"

    while (epoch < n_epochs):        
        epoch = epoch + 1
        print "Starting***"
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):

                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):

                cost_epoch = train_model(minibatch_index)
                # cost_epoch = train_model_user(minibatch_index)
                set_zero(zero_vec)
        train_newx = train_pred_model(train_set_x_org)
        [val_newx, ypred] = val_model(val_set_x_org)
        val_perf = avg_fscore(ypred, val_set_y_org)
        [test_newx, ypred] = test_model(test_set_x)
        test_perf = avg_fscore(ypred, test_set_y)
        print test_perf
        if val_perf >= best_val_perf:
            newxx = [train_newx, val_newx, test_newx]
            best_val_perf = val_perf

        # SVM training and testing
        regs = [0.05, 0.1, 0.5, 1.0]
        for reg in regs:
             clf = svm.LinearSVC(C=reg, loss='l1', tol=1e-4)
             clf = clf.fit(train_newx, train_set_y_org)
             ypred = clf.predict(train_newx)
             train_perf = avg_fscore(ypred, train_set_y_org)
             ypred = clf.predict(val_newx)
             val_perf = avg_fscore(ypred, val_set_y_org)
             ypred = clf.predict(test_newx)
             test_perf = avg_fscore(ypred, test_set_y)
             print('Linear -> reg %.2f, %%, val perf %f, test perf %f' % (reg,  val_perf*100., test_perf*100.))
             result.write('Linear -> reg %.2f, %%, val perf %f, test perf %f' % (reg,  val_perf*100., test_perf*100.))
    return newxx

def user_tweet_cost(layer1_input):
    print "Testing for my cost function******"
    a , b =  layer1_input.shape
    print layer1_input.shape
    return 0
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


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=50, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=50, k=300, filter_h=5, val_test_splits=[2,3]):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    usertrain, userval, usertest = [], [], []
    val_split, test_split = val_test_splits
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        # print rev["split"]
      #  if rev["user"] == 'unknown':
       #     rev["user"] =0
        sent.append(rev["y"])
        if rev["split"]==val_split:            
            val.append(sent)
            userval.append(rev["user"])
        elif rev["split"]==test_split:  
            test.append(sent)
            usertest.append(rev["user"])
        else:
            train.append(sent)
            usertrain.append(rev["user"])


    train = np.array(train,dtype="int")
    val = np.array(val,dtype="int")
    test = np.array(test,dtype="int")
    img_h = len(train[0])-1
    usertrain = np.array(usertrain,dtype="int")
    userval = np.array(userval,dtype="int")
    usertest = np.array(usertest,dtype="int")
    return [train, val, test, usertrain, userval, usertest]
 
def get_cnn_representation(revs, U, word_idx_map, max_l, filter_h=5, val_test_splits=[2,3]):
    k = len(U[0])
    datasets = make_idx_data(revs, word_idx_map, max_l=max_l, k=k, filter_h=filter_h, val_test_splits=val_test_splits)
    cnn_representation = train_conv_net(datasets,
                          U,
                          img_w=k,
                          lr_decay=0.95,
                          filter_hs=[3,4,5],
                          conv_non_linear="tanh",
                          hidden_units=[100,2], 
                          shuffle_batch=True, 
                          n_epochs=25, 
                          sqr_norm_lim=9,
                          non_static=True,
                          batch_size=50,
                          dropout_rate=[0.5])
    return cnn_representation

if __name__=="__main__":
    print "loading data...",
    #x = cPickle.load(open("mr.p","rb"))
    #x = cPickle.load(open("sst2.p","rb"))
    x = cPickle.load(open("semeval.pkl","rb"))
    revs, W, word_idx_map, max_l , UserW, userDictionary,user_graph = x[0], x[1], x[2],x[3],x[4],x[5],x[6]
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    # W =word_vec.W
    W2=None
    # word_idx_map =  word_vec.word_idx_map

    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    k = len(U[0])
    # train/val/test results
    datasets = make_idx_data(revs, word_idx_map, max_l=max_l, k=k, filter_h=5)

    perf = train_conv_net(datasets,
                          U,UserW, userDictionary,user_graph,
                          img_w=k,
                          lr_decay=0.95,
                          filter_hs=[3,4,5],
                          conv_non_linear="tanh",
                          hidden_units=[100,2], 
                          shuffle_batch=False,
                          n_epochs=30,
                          sqr_norm_lim=9,
                          non_static=non_static,
                          batch_size=50,
                          dropout_rate=[0.5])
    print "perf: " + str(perf)

