import numpy as np
from collections import defaultdict
import sys, re, logging, argparse
import langid
from twokenize import tokenize

logger = logging.getLogger("customized.twsent.procraw")

def process_semeval(data_folder, outfname, idfname):
    """
    Process dataset of SemEval 2013 task 2 subtask B
    Only keep postive and negative tweets
    Format: tweet|label,user_id,split\n
    """
    logger.info("start processing tweets for SemEval 2013 task 2 subtask B")
    tid_uid_map = {}
    with open(idfname, "rb") as f:
        for line in f:  
            parts = line.strip().split()
            tid_uid_map[parts[0]] = parts[1]

    train_file, val_file, test_file = data_folder[0], data_folder[1], data_folder[2]
    fout = open(outfname, "w")
    with open(train_file, "rb") as f:
        for line in f:  
            parts = line.strip().split("\t")
            label = parts[2]
            tweet = parts[3]
            userid = "unknown"
            if parts[0] in tid_uid_map: userid = tid_uid_map[parts[0]]
            if tweet == "Not Available": continue
            tweet = clean_tweet(tweet)
            tweet = clean_tweet_toks(tokenize(tweet))
            if label == "positive":
                fout.write(tweet + "|1," + userid + ",1\n")
            elif label == "negative":
                fout.write(tweet + "|0," + userid + ",1\n")
    with open(val_file, "rb") as f:
        for line in f:  
            parts = line.strip().split("\t")
            label = parts[2]
            tweet = parts[3]
            userid = "unknown"
            if parts[0] in tid_uid_map: userid = tid_uid_map[parts[0]]
            if tweet == "Not Available": continue
            tweet = clean_tweet(tweet)
            tweet = clean_tweet_toks(tokenize(tweet))
            if label == "positive":
                fout.write(tweet + "|1," + userid + ",2\n")
            elif label == "negative":
                fout.write(tweet + "|0," + userid + ",2\n")
    with open(test_file, "rb") as f:
        for line in f:  
            parts = line.strip().split("\t")
            label = parts[2]
            tweet = parts[3]
            userid = "unknown"
            if parts[0] in tid_uid_map: userid = tid_uid_map[parts[0]]
            if tweet == "Not Available": continue
            tweet = clean_tweet(tweet)
            tweet = clean_tweet_toks(tokenize(tweet))
            if label == "positive":
                fout.write(tweet + "|1," + userid + ",3\n")
            elif label == "negative":
                fout.write(tweet + "|0," + userid + ",3\n")
    fout.close()
    logger.info("finish processing data")

def process_omd(fname, outfname, idfname):
    """
    Process OMD dataset: each tweet is associated with three votings which 1:neg 2:pos
    Only keep postive and negative tweets according two 2/3 agreement
    Format: tweet|label\n
    """
    logger.info("start processing tweets for OMD")
    tid_uid_map = {}
    with open(idfname, "rb") as f:
        for line in f:  
            parts = line.strip().split()
            tid_uid_map[parts[0]] = parts[1]

    fout = open(outfname, "w")
    with open(fname, "rb") as f:
        for line in f:  
            parts = line.strip().split("\t")
            votes = [int(parts[-1]), int(parts[-2]), int(parts[-3])]
            votes.sort()
            if votes[0] == 1 and votes[1] == 1: label = "0"
            elif votes[1] == 2 and (votes[0] == 2 or votes[2] == 2): label = "1"
            else: continue
            tweet = parts[2]
            userid = "unknown"
            if parts[0] in tid_uid_map: userid = tid_uid_map[parts[0]]
            if tweet[0] == "\"": tweet = tweet[1:].strip()
            if tweet[-1] == "\"": tweet = tweet[:-1].strip()
            tweet = clean_tweet(tweet)
            tweet = clean_tweet_toks(tokenize(tweet))
            fout.write(tweet + "|" + label +  "," + userid + "\n")
    fout.close()
    logger.info("finish processing data")


def clean_tweet(tweet):
    """
    clean HTML tags
    """
    tweet = re.sub(r"&lt;", ">", tweet)
    tweet = re.sub(r"&gt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
    tweet = re.sub(r"&nbsp;", " ", tweet)
    return tweet


def clean_tweet_toks(toks):
    cltoks = []
    for tok in toks:
        tok = re.sub(r"@.+", "@USER", tok)
        #tok = re.sub(r"#.+", "#HASHTAG", tok)
        tok = re.sub(r"http.+", "#@URL", tok)
        #if tok.startswith("#"): tok = tok[1:]
        cltoks.append(tok)
    return u" ".join(cltoks).encode('utf-8')

def filter_nonEN_tweets(in_file, out_file, field):
    f_out = open(out_file, "a")
    with open(in_file, "rb") as f:
        for line in f:  
            parts = line.strip().split("\t")
            tweet = parts[field]
            if langid.classify(tweet)[0] != "en": continue
            try :
                tweet = clean_tweet(tweet)
                tweet = clean_tweet_toks(tokenize(tweet))
            except :
                continue
                pass
            f_out.write(tweet + "\n")
    f_out.close()


if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="name of dataset: omd or semeval")
    args = parser.parse_args()

    if args.dataset_name == "omd":
        fname = "../data/twitter-sentiment/OMD/omd.tsv"
        outfname = "../data/twitter-sentiment/OMD/omd.txt"
        idfname = "../data/twitter-sentiment/OMD/omd-tids-uids.txt"
        process_omd(fname, outfname, idfname)
    elif args.dataset_name == "semeval":
        input_dir = "../data/twitter-sentiment/SemEval"
        outfname = "../data/twitter-sentiment/SemEval/semeval.txt"
        idfname = "../data/twitter-sentiment/SemEval/semeval-tids-uids.txt"
        data_folder = [input_dir + "/twitter-train-cleansed-B.out", input_dir + "/twitter-dev-gold-B.out", input_dir + "/twitter-test-GOLD-B.tsv"]    
        process_semeval(data_folder, outfname, idfname)
    else:
        print "only omd or semeval is supported for now"
    logger.info('end logging')

