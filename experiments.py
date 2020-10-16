import random

import nltk
import re
import pandas
import warnings
import gensim
import numpy as np
from info_gain import info_gain
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn import svm, naive_bayes, ensemble, tree, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')


def structure():
    # Specify the file path of the dataset
    file_path = "1.Training data for classifier.xlsx"

    data = pandas.read_excel(file_path, 0, usecols=[1, 2, 3], names=['text', 'type', 'label'])

    return data


def text_preprocess(data):
    tokenizer = nltk.WordPunctTokenizer()
    porter_stemmer = PorterStemmer()
    stop_words = set(stopwords.words('English'))
    comp = re.compile('[^A-Z^a-z ]')

    for i in range(len(data)):
        data_item = data[i]

        if data_item is None:
            x = 1

        # Clean by a regular expression
        data_item = comp.sub('', data_item)

        # Spilt into words
        data_item = tokenizer.tokenize(str(data_item).lower())

        # Remove stop words
        data_item = [word for word in data_item if word not in stop_words]

        # Stemming
        for w_idx in range(len(data_item)):
            data_item[w_idx] = porter_stemmer.stem(data_item[w_idx])

        data[i] = ' '.join(data_item)

    return data

# using bow
def bow(data):
    # We consider words that appear more than once, so we set the parameter min_df=2
    bow_vector = CountVectorizer(min_df=2)

    data_bow = bow_vector.fit_transform(data)

    return data_bow


def tf_idf(data):
    # We consider words that appear more than once, so we set the parameter min_df=2
    tf_idf_vector = TfidfVectorizer(min_df=2)

    data_tf_idf = tf_idf_vector.fit_transform(data)

    return data_tf_idf


def word2vec(data):
    items = []

    for s in data:
        items.append(str(s).split(' '))

    vec_size = 150

    model = gensim.models.word2vec.Word2Vec(items, min_count=2, size=vec_size)

    word2vec_data = []

    for s in items:
        v = np.zeros(vec_size)
        count = 0

        for word in s:
            try:
                count += 1
                v += model[word]
            except KeyError:
                continue
        v /= count
        word2vec_data.append(v)

    return word2vec_data


def training_dic(label):
    manually_extracted_terms = ['heartbeat', 'ping', 'ping/echo', 'beat', 'decorator', 'piggybacking', 'outbound',
                                'period', 'audit', 'trail', 'wizard', 'log', 'string', 'category', 'thread', 'pooling',
                                'pool', 'thread', 'connect', 'sparrow', 'processor', 'worker', 'time-wait', 'prototype',
                                'singleton', 'strategy', 'chain of responsibility', 'lazy load', 'static scheduling',
                                'dynamic priority scheduling', 'authentic', 'credential', 'challenge', 'login', 'FIFO',
                                'fixed-priority', 'dynamic priority scheduling', 'schedule', 'task', 'priority',
                                'adaptor', 'bridge', 'composite', 'flyweight', 'memento', 'observer', 'proxy',
                                'strategy', 'checkpoint', 'checkpoints', 'barrier', 'weak point', 'layoff', 'restraint',
                                'austerity', 'abridgement', 'deliver', 'spare', 'unoccupied', 'option', 'unused',
                                'logging', 'minutes', 'redundancy replication', 'redundancy storage', 'zone-redundant',
                                'geo-redundant', 'replication', 'voting', 'vote', 'balloting', 'choosing', 'voter',
                                'processor', 'preferred', 'shadow operation', 'shadow mode', 'secure session',
                                'security', 'removal', 'time out', 'run out', 'constraint', 'action', 'monitor',
                                'timer', 'runtime', 'time stamp', 'timestamp', 'time strap', 'sanity checking',
                                'sanity check', 'functional redundancy', 'function requirement allocation', 'parallel',
                                'separate', 'warm restart', 'dual redundancy', 'resisting attacks', 'detecting',
                                'detect', 'recovering', 'recover', 'sensor', '', 'authenticate', 'confidentiality',
                                'exposure', 'limit access', 'passwords', 'one-time', 'passwords',
                                'digital certificates', 'maintain data confidentiality', 'handle', 'protecting',
                                'routine', 'storage', 'mandatory', 'recovering from attacks', 'state', 'maintain',
                                'maintaining', 'redundant', 'access control', 'profile', 'performance',
                                'processing_time', 'response_time', 'resource_consumption', 'throughput', 'efficiency',
                                'carrying_into_action', 'carrying_out', 'operation', 'achievement', 'interaction',
                                'accomplishment', 'action', 'maintainability', 'update', 'modify', 'modular',
                                'decentralized', 'encapsulation', 'dependency', 'interdependent', 'interdependent',
                                'understandability', 'modifiability', 'modularity', 'maintain', 'analyzability',
                                'changeability', 'testability', 'encapsulation', 'compatibility', 'co-existence',
                                'interoperability', 'exchange', 'sharing', 'usability', 'flexibility', 'interface',
                                'user-friendly', 'default', 'configure', 'serviceability', 'convention',
                                'accessibility', 'gui', 'serviceableness', 'useableness', 'utility', 'useable',
                                'learnability', 'understandability', 'operability', 'function', 'use', 'reliability',
                                'failure', 'bug', 'resilience', 'crash', 'stability', 'dependable', 'dependability',
                                'irresponsibleness', 'recover', 'recoverability', 'tolerance', 'error', 'fails',
                                'redundancy', 'integrity', 'irresponsibleness', 'dependable', 'maturity',
                                'recoverability', 'accountability', 'answerableness', 'functional', 'function',
                                'accuracy', 'completeness', 'suitability', 'compliance', 'performing', 'employable',
                                'functionality', 'complexity', 'functioning', 'security', 'safe', 'vulnerability',
                                'trustworthy', 'firewall', 'login', 'password', 'pin', 'auth', 'verification',
                                'protection', 'certificate', 'security_system', 'law', 'portability', 'portable',
                                'cross_platform', 'transfer', 'transformability', 'documentation', 'standardized',
                                'migration', 'specification', 'movability', 'moveableness', 'replaceability',
                                'adaptability']

    manually_extracted_terms = set(manually_extracted_terms)

    # Load architecture posts from SO
    architecture_posts = pandas.read_excel()

    # preprocess
    pre_arch_posts = text_preprocess(architecture_posts)

    # training by Word2Vec
    items = []

    for s in pre_arch_posts:
        items.append(str(s).split(' '))

    model = gensim.models.word2vec.Word2Vec(items, min_count=2, size=150)
    automatic_extracted_terms = set()

    for w in manually_extracted_terms:
        if w in model:
            for t in model.wv.similar_by_word(w):
                if t[1] > 0.350:
                    automatic_extracted_terms.add(t[0])

    bow_vector = CountVectorizer()
    bow_arch_data = bow_vector.fit_transform(pre_arch_posts).toarray()

    for t in automatic_extracted_terms:
        # find term index in bow vocabulary
        index = bow_vector.vocabulary_.get(t, -1)

        if index != -1:
            # filtering terms by calculating information gain ratio
            igr = info_gain.info_gain_ratio(bow_vector[:, index], label)

            if igr < 0.300:
                automatic_extracted_terms.remove(t)

    final_terms = manually_extracted_terms | automatic_extracted_terms


def train_and_evaluation(technique, name, model, training_data, test_data, training_label, test_label, report):
    model.fit(training_data, training_label)

    pre_results = model.predict(test_data)

    # Evaluate classifiers by weighted precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(test_label, pre_results, labels=[0, 1]).ravel()

    precision = precision_score(test_label, pre_results)
    recall = recall_score(test_label, pre_results)
    f1 = f1_score(test_label, pre_results)

    if report:
        print("feature extraction technique: " + technique + "  ML:" + name)
        print("tp: " + str(tp) + " tn: " + str(tn) + " fp: " + str(fp) + " fn: " + str(fn))
        print("precision: %.3f" % precision)
        print("recall: %.3f" % recall)
        print("f1-score: %.3f" % f1)
        print("\n")

    return precision, recall, f1


def classifiers(feature_extraction_technique, data, label, report):
    training_data, test_data, training_label, test_label = train_test_split(data, label, test_size=0.3,
                                                                            random_state=random.randint(0, 100))

    # Six machine learning classifiers
    ml_classifiers = [
        ('SVM', svm.SVC(C=1.0, kernel='rbf', random_state=0)),
        ('Bayes', naive_bayes.BernoulliNB(alpha=1.0)),
        ('DT', tree.DecisionTreeClassifier(criterion="gini", random_state=0)),
        ('LR', linear_model.LogisticRegression(C=1.0, random_state=0)),
        ('Bagging', ensemble.BaggingClassifier(random_state=0)),
        ('RF', ensemble.RandomForestClassifier(criterion="gini", random_state=0)),
    ]

    classification_results = {}

    for name, model in ml_classifiers:
        precision, recall, f1 = train_and_evaluation(feature_extraction_technique, name, model, training_data,
                                                     test_data, training_label,
                                                     test_label,
                                                     report)

        classification_results[name] = {'precision': precision, 'recall': recall, 'f1': f1}

    return classification_results


def experiment():
    # Input data
    dataset = structure()

    data = dataset['text'].tolist()
    label = dataset['label'].tolist()

    # Preprocess data
    data = text_preprocess(data)

    # Feature extraction techniques
    feature_extraction_techniques = [
        ('BoW', bow(data)),
        ('TF-IDF', tf_idf(data)),
        ('Word2Vec', word2vec(data)),
    ]

    for technique, data in feature_extraction_techniques:
        # run 10 times and get average results
        classification_results = {'SVM': {'precision': [], 'recall': [], 'f1': []},
                                  'Bayes': {'precision': [], 'recall': [], 'f1': []},
                                  'DT': {'precision': [], 'recall': [], 'f1': []},
                                  'LR': {'precision': [], 'recall': [], 'f1': []},
                                  'Bagging': {'precision': [], 'recall': [], 'f1': []},
                                  'RF': {'precision': [], 'recall': [], 'f1': []},
                                  }

        for i in range(0, 10):
            res = classifiers(technique, data, label, False)

            for model in dict(res).keys():
                for v in dict(res[model]).keys():
                    classification_results[model][v].append(res[model][v])

        # report average results
        for model in classification_results.keys():
            print("average results: feature extraction technique: " + technique + "  ML:" + model)
            print("precision: %.3f" % np.mean(classification_results[model]['precision']))
            print("recall: %.3f" % np.mean(classification_results[model]['recall']))
            print("f1-score: %.3f" % np.mean(classification_results[model]['f1']))
            print("\n")


if __name__ == "__main__":
    # Run this function to get our experiments results
    experiment()
