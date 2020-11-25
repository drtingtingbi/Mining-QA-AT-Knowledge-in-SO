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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')


def structure():
    # Specify the file path of the dataset
    file_path = "../dataset/Training data.xlsx"

    data = pandas.read_excel(file_path, 0, usecols=[1, 2, 3], names=['text', 'type', 'label'])

    return data


def text_preprocess(data):
    tokenizer = nltk.WordPunctTokenizer()
    porter_stemmer = PorterStemmer()
    stop_words = set(stopwords.words('English'))
    comp = re.compile('[^A-Z^a-z ]')

    for i in range(len(data)):
        data_item = data[i]

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


def training_dic(QA_AT_data, QA_AT_label):
    manually_extracted_term_lists = ['heartbeat', 'ping', 'ping/echo', 'beat', 'decorator', 'piggybacking', 'outbound',
                                     'period', 'audit', 'trail', 'wizard', 'log', 'string', 'category', 'thread',
                                     'pooling',
                                     'pool', 'thread', 'connect', 'sparrow', 'processor', 'worker', 'time-wait',
                                     'prototype',
                                     'singleton', 'strategy', 'chain of responsibility', 'lazy load',
                                     'static scheduling',
                                     'dynamic priority scheduling', 'authentic', 'credential', 'challenge', 'login',
                                     'FIFO',
                                     'fixed-priority', 'dynamic priority scheduling', 'schedule', 'task', 'priority',
                                     'adaptor', 'bridge', 'composite', 'flyweight', 'memento', 'observer', 'proxy',
                                     'strategy', 'checkpoint', 'checkpoints', 'barrier', 'weak point', 'layoff',
                                     'restraint',
                                     'austerity', 'abridgement', 'deliver', 'spare', 'unoccupied', 'option', 'unused',
                                     'logging', 'minutes', 'redundancy replication', 'redundancy storage',
                                     'zone-redundant',
                                     'geo-redundant', 'replication', 'voting', 'vote', 'balloting', 'choosing', 'voter',
                                     'processor', 'preferred', 'shadow operation', 'shadow mode', 'secure session',
                                     'security', 'removal', 'time out', 'run out', 'constraint', 'action', 'monitor',
                                     'timer', 'runtime', 'time stamp', 'timestamp', 'time strap', 'sanity checking',
                                     'sanity check', 'functional redundancy', 'function requirement allocation',
                                     'parallel',
                                     'separate', 'warm restart', 'dual redundancy', 'resisting attacks', 'detecting',
                                     'detect', 'recovering', 'recover', 'sensor', 'authenticate', 'confidentiality',
                                     'exposure', 'limit access', 'passwords', 'one-time', 'passwords',
                                     'digital certificates', 'maintain data confidentiality', 'handle', 'protecting',
                                     'routine', 'storage', 'mandatory', 'recovering from attacks', 'state', 'maintain',
                                     'maintaining', 'redundant', 'access control', 'profile', 'performance',
                                     'processing_time', 'response_time', 'resource_consumption', 'throughput',
                                     'efficiency',
                                     'carrying_into_action', 'carrying_out', 'operation', 'achievement', 'interaction',
                                     'accomplishment', 'action', 'maintainability', 'update', 'modify', 'modular',
                                     'decentralized', 'encapsulation', 'dependency', 'interdependent', 'interdependent',
                                     'understandability', 'modifiability', 'modularity', 'maintain', 'analyzability',
                                     'changeability', 'testability', 'encapsulation', 'compatibility', 'co-existence',
                                     'interoperability', 'exchange', 'sharing', 'usability', 'flexibility', 'interface',
                                     'user-friendly', 'default', 'configure', 'serviceability', 'convention',
                                     'accessibility', 'gui', 'serviceableness', 'useableness', 'utility', 'useable',
                                     'learnability', 'understandability', 'operability', 'function', 'use',
                                     'reliability',
                                     'failure', 'bug', 'resilience', 'crash', 'stability', 'dependable',
                                     'dependability',
                                     'irresponsibleness', 'recover', 'recoverability', 'tolerance', 'error', 'fails',
                                     'redundancy', 'integrity', 'irresponsibleness', 'dependable', 'maturity',
                                     'recoverability', 'accountability', 'answerableness', 'functional', 'function',
                                     'accuracy', 'completeness', 'suitability', 'compliance', 'performing',
                                     'employable',
                                     'functionality', 'complexity', 'functioning', 'security', 'safe', 'vulnerability',
                                     'trustworthy', 'firewall', 'login', 'password', 'pin', 'auth', 'verification',
                                     'protection', 'certificate', 'security_system', 'law', 'portability', 'portable',
                                     'cross_platform', 'transfer', 'transformability', 'documentation', 'standardized',
                                     'migration', 'specification', 'movability', 'moveableness', 'replaceability',
                                     'adaptability']

    manually_extracted_terms = set(manually_extracted_term_lists)

    # Load architecture posts from SO
    architecture_posts = pandas.read_excel('../dataset/Architecture posts - Stack Overflow.xlsx', 0, usecols=[1, 5, 6],
                                           names=['title', 'question', 'answers'])
    arch_items = []
    arch_terms = set()

    for i in range(len(architecture_posts)):
        arch_items.append(
            str(architecture_posts['title'][i]) + ' ' + str(architecture_posts['question'][i]) + ' ' + str(
                architecture_posts['answers'][i]))

    for i in range(len(text_preprocess(arch_items))):
        arch_items[i] = str(arch_items[i]).split(' ')

        for w in arch_items[i]:
            arch_terms.add(w)

    QA_AT_items = []

    for i in range(len(QA_AT_data)):
        if QA_AT_label[i] == 1:
            QA_AT_items.append(str(QA_AT_data[i]).split(' '))

    # training by Word2Vec
    items = arch_items + QA_AT_items

    word2vec_model = gensim.models.word2vec.Word2Vec(items, size=150)
    automatic_extracted_terms = set()

    for w in manually_extracted_terms:
        if w in word2vec_model:
            for t in word2vec_model.wv.similar_by_word(w):
                # retain terms that are in architecture posts and have a similarity greater than 0.35
                if t[1] > 0.35 and t[0] in arch_terms:
                    automatic_extracted_terms.add(t[0])

    bow_vector = CountVectorizer()
    bow_QA_AT_data = bow_vector.fit_transform(QA_AT_data).todense()

    final_training_dictionary = set()
    v_len = bow_QA_AT_data.shape[0]

    for t in automatic_extracted_terms:
        # find current term's index in bow vocabulary
        index = bow_vector.vocabulary_.get(t, -1)

        if index != -1:
            # represent the distribution of current term in QA_AT data
            v = np.reshape(bow_QA_AT_data[:, index], v_len).tolist()[0]
            igr = info_gain.info_gain_ratio(v, QA_AT_label)

            if igr > 0.30:
                # retain terms in QA_AT data and have an information gain ratio greater than 0.30
                final_training_dictionary.add(t)

    final_training_dictionary = manually_extracted_terms | final_training_dictionary

    for t in final_training_dictionary:
        print(t)

    return final_training_dictionary, word2vec_model


def argument():
    x = 1


def evaluation(technique, name, model, data, label, k_cross_fold, report):
    pre_results = cross_val_predict(model, data, label, cv=k_cross_fold)

    tn, fp, fn, tp = confusion_matrix(label, pre_results).ravel()

    # Evaluate classifiers by weighted precision, recall, and F1-score
    precision = precision_score(label, pre_results)
    recall = recall_score(label, pre_results)
    f1 = f1_score(label, pre_results)

    if report:
        print("feature extraction technique: " + technique + "  ML:" + name)
        print("TP: " + str(tp) + " FP: " + str(fp) + " TN: " + str(tn) + " FN: " + str(fn))
        print("precision: %.3f" % precision)
        print("recall: %.3f" % recall)
        print("f1-score: %.3f" % f1)
        print("\n")

        # print(classification_report(label, pre_results, digits=3))


def train_classifiers(feature_extraction_technique, data, label, report):
    # 10-fold-cross validation
    k_cross_fold = KFold(n_splits=10, shuffle=False, random_state=None)

    # Six machine learning classifiers
    classifiers = [
        ('SVM', svm.SVC(C=1.0, kernel='rbf', random_state=0)),
        ('Bayes', naive_bayes.BernoulliNB(alpha=1.0)),
        ('DT', tree.DecisionTreeClassifier(criterion="gini", random_state=0)),
        ('LR', linear_model.LogisticRegression(C=1.0, random_state=0)),
        ('Bagging', ensemble.BaggingClassifier(random_state=0)),
        ('RF', ensemble.RandomForestClassifier(criterion="gini", random_state=0)),
    ]

    for name, model in classifiers:
        evaluation(feature_extraction_technique, name, model, data, label, k_cross_fold, report)


def experiment():
    # Input data
    dataset = structure()

    data = dataset['text'].tolist()
    label = dataset['label'].tolist()

    # Preprocess data
    data = text_preprocess(data)

    training_dic(data, label)

    # Feature extraction techniques
    # feature_extraction_techniques = [
    #     ('BoW', bow(data)),
    #     # ('TF-IDF', tf_idf(data)),
    #     # ('Word2Vec', word2vec(data)),
    # ]
    #
    # for technique, data in feature_extraction_techniques:
    #     train_classifiers(technique, data, label, True)


if __name__ == "__main__":
    # Run this function to get our experiments results
    experiment()
    
    
