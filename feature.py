from collections import defaultdict
import sys, re, os

class Feature(object):
    """
    Extract features for twitter sentiment classification task
    Yi Yang (yangyiycc@gmail.com)
    """
    def __init__(self):
        # resources for extracting features
        cmutagger_path = "../data/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar"
        nrc_emotion_path = "../data/Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon.txt"
        bing_path = "../data/Lexicons/opinion-lexicon-English/bing-lexicon.txt"
        hashtag_sentiment_path_uni = "../data/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt"
        hashtag_sentiment_path_bi = "../data/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/bigrams-pmilexicon.txt"
        sentiment140_path_uni = "../data/Lexicons/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt"
        sentiment140_path_bi = "../data/Lexicons/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt"
        brown_cluster_path = "../data/50mpaths2"
    
        self.nrc_emotion_dict, self.bing_dict, self.hashtag_sentiment_dict, self.sentiment140_dict = {}, {}, {}, {}
        self.brown_cluster_dict = {}
        with open(nrc_emotion_path, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if parts[2] == "0": continue
                key = parts[0].lower()
                if key not in self.nrc_emotion_dict:
                    self.nrc_emotion_dict[key] = []
                self.nrc_emotion_dict[key].append(parts[1])
        with open(bing_path, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                key = parts[0].lower()
                self.bing_dict[key] = parts[1]
        with open(hashtag_sentiment_path_uni, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if parts[2] == "0": continue
                key = parts[0].lower()
                self.hashtag_sentiment_dict[key] = float(parts[1])
        with open(hashtag_sentiment_path_bi, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if parts[2] == "0": continue
                key = parts[0].lower()
                self.hashtag_sentiment_dict[key] = float(parts[1])
        with open(sentiment140_path_uni, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if parts[2] == "0": continue
                key = parts[0].lower()
                self.sentiment140_dict[key] = float(parts[1])
        with open(sentiment140_path_bi, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if parts[2] == "0": continue
                key = parts[0].lower()
                self.sentiment140_dict[key] = float(parts[1])
        with open(brown_cluster_path, "rb") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if len(parts) < 2: continue
                self.brown_cluster_dict[parts[1]] = parts[0]
    
    
    def NRC_feature_extractor(self, tweet):
        features = {}
        "case senstive features"
    #    all_caps, hashtags = 0, 0
    #    toks = tweet.strip().split()
    #    for tok in toks:
    #        if tok == tok.upper(): all_caps += 1
    #        if tok.startswith("#"): hashtags += 1
    #    features["all_caps"] = all_caps
    #    features["hashtags"] = hashtags
        # POS tags
    #    with open("tmp.pos", "w") as f: 
    #        f.write(tweet)
    #    os.system("java -XX:ParallelGCThreads=2 -Xmx500m -jar %s tmp.pos > tmp.tag" %cmutagger_path)
    #    with open("tmp.tag", "rb") as f:
    #        for line in f:
    #            if line.strip() == "": continue
    #            parts = line.split("\t")
    #            tags = parts[1].split()
    #            for tag in tags:
    #                if "POS_"+tag not in features: features["POS_"+tag] = 0
    #                features["POS_"+tag] += 1
    
        "case insenstive features"
        tweet = tweet.lower()
        # character ngram : 3,4,5
    #    for n in range(len(tweet)-2):
    #        features["char_ngram_"+tweet[n:n+3]] = 1
    #        if n < len(tweet)-3: features["char_ngram_"+tweet[n:n+4]] = 1
    #        if n < len(tweet)-4: features["char_ngram_"+tweet[n:n+5]] = 1
        # Lexicon features: nrc_emotion_dict, bing_dict, hashtag_sentiment_dict, sentiment140_dict
        toks = tweet.strip().split()
        for n,tok in enumerate(toks):
            # 1. NRC Emotion
            if tok in self.nrc_emotion_dict:
                for emo in self.nrc_emotion_dict[tok]:
                    if "nrc_emotion_" + emo not in features: features["nrc_emotion_" + emo] = 0
                    features["nrc_emotion_" + emo] += 1
            # 2. Bing Liu        
            if tok in self.bing_dict:
                emo = self.bing_dict[tok]
                if "bing_" + emo not in features: features["bing_" + emo] = 0
                features["bing_" + emo] += 1
            # 3. Hashtag Sentiment
            features["hashtag_sentiment_uni_count"] = 0
            features["hashtag_sentiment_uni_score"] = 0.0
            features["hashtag_sentiment_uni_max"] = 0.0
            features["hashtag_sentiment_uni_last"] = 0.0
            features["hashtag_sentiment_bi_count"] = 0
            features["hashtag_sentiment_bi_score"] = 0.0
            features["hashtag_sentiment_bi_max"] = 0.0
            features["hashtag_sentiment_bi_last"] = 0.0
            if tok in self.hashtag_sentiment_dict:
                score = self.hashtag_sentiment_dict[tok]
                features["hashtag_sentiment_uni_score"] += score
                features["hashtag_sentiment_uni_last"] = score
                if score > features["hashtag_sentiment_uni_max"]: features["hashtag_sentiment_uni_max"] = score
                if score > 0: features["hashtag_sentiment_uni_count"] += 1
            if n < len(toks)-1 and toks[n]+" "+toks[n+1] in self.hashtag_sentiment_dict:
                tok = toks[n] + " " + toks[n+1]
                score = self.hashtag_sentiment_dict[tok]
                features["hashtag_sentiment_bi_score"] += score
                features["hashtag_sentiment_bi_last"] = score
                if score > features["hashtag_sentiment_bi_max"]: features["hashtag_sentiment_bi_max"] = score
                if score > 0: features["hashtag_sentiment_bi_count"] += 1
            # 4. Sentiment140
            features["sentiment140_uni_count"] = 0
            features["sentiment140_uni_score"] = 0.0
            features["sentiment140_uni_max"] = 0.0
            features["sentiment140_uni_last"] = 0.0
            features["sentiment140_bi_count"] = 0
            features["sentiment140_bi_score"] = 0.0
            features["sentiment140_bi_max"] = 0.0
            features["sentiment140_bi_last"] = 0.0
            if tok in self.sentiment140_dict:
                score = self.sentiment140_dict[tok]
                features["sentiment140_uni_score"] += score
                features["sentiment140_uni_last"] = score
                if score > features["sentiment140_uni_max"]: features["sentiment140_uni_max"] = score
                if score > 0: features["sentiment140_uni_count"] += 1
            if n < len(toks)-1 and toks[n]+" "+toks[n+1] in self.hashtag_sentiment_dict:
                tok = toks[n] + " " + toks[n+1]
                score = self.hashtag_sentiment_dict[tok]
                features["sentiment140_bi_score"] += score
                features["sentiment140_bi_last"] = score
                if score > features["sentiment140_bi_max"]: features["sentiment140_bi_max"] = score
                if score > 0: features["sentiment140_bi_count"] += 1
    
        return features
    
