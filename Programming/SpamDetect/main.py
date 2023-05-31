import argparse
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from math import log, floor
from numpy.linalg import norm
from time import time
from abc import ABC, abstractmethod

nltk.download("punkt")
nltk.download("stopwords")

model_global = []
precision_global = []
recall_global = []
Fscore_global = []
accuracy_global = []
time_global = []

def main():
    # to ensure reproducibility
    np.random.seed(1337)

    parser = argparse.ArgumentParser(description="Binary classifiers.")
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Shows the occurence of words as a wordcloud",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="trains and tests the algorithms and gives results in différent metrics",
    )
    parser.add_argument(
        "-c",
        "--classify",
        type=str,
        help="Classifies the given text into spam or not spam using TFxIDF",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Select the dataset among spam, clickbait and reviews",
        default="spam"
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        type=str,
        help="Select the model(s) to run among random, bow, naive, knn, knn-cwc, svm, svm-cwc and ga",
        default=["bow", "naive"]
    )
    args = parser.parse_args()

    # Load data
    if args.data == "spam":
        data, data_type = load_spam_data()
        name_res = "results.spam.csv"
    elif args.data == "clickbait":
        data, data_type = load_clickbait_data()
        name_res = "results.clickbait.csv"
    else:
        data, data_type = load_rt_reviews_data()
        name_res = "results.rt_reviews.csv"

    # On va maintenant réaliser les deux sous-ensembles que sont
    # Le training set (75% des données) et le Testing Set (25% des données).
    trainIndex, testIndex = list(), list()
    for i in range(data.shape[0]):
        if np.random.uniform(0, 1) < 0.75:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = data.loc[trainIndex]
    testData = data.loc[testIndex]

    # On exécute un reset des index dans les deux sous-ensembles.
    trainData.reset_index(inplace=True)
    trainData.drop(["index"], axis=1, inplace=True)
    testData.reset_index(inplace=True)
    testData.drop(["index"], axis=1, inplace=True)

    # Retrieve models
    models = {}
    for model in args.models:
        model_global.append(model)
        if model == "random":
            models["Random"] = RandomClassifier(trainData)
        elif model == "bow":
            models["Bow"] = BowClassifier(trainData)
        elif model == "naive":
            models["TF x IDF"] = TFIDFClassifier(trainData)
        elif model == "knn":
            models["KNN"] = KNNClassifier(trainData)
        elif model == "knn-cwc":
            models["KNN with Confidence Weight classifier"] = KNNConfWeightClassifier(trainData)
        elif model == "svm":
            models["SVM"] = SVMClassifier(trainData)
        elif model == "svm-cwc":
            models["SVM with Confidence weigh score"] = SVMConfWeightClassifier(trainData)
        elif model == "ga":
            models["GA"] = GAClassifier(trainData)
    # On peut visualiser les mots clés les plus fréquents des spams et
    # faire la même chose pour les non-spams.
    if args.show:
        spam_words = " ".join(list(data[data["label"] == 1]["message"]))
        spam_wc = WordCloud(width=512, height=512).generate(spam_words)
        plt.figure(figsize=(10, 8), facecolor="k")
        plt.imshow(spam_wc)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
    elif args.test:
        msg_to_classify = ["I cant pick the phone right now. Pls send a message",
                           "Congratulations ur awarded $500 "]
        for k in models:
            train_and_predict(models[k], k, data_type, msg_to_classify, testData)

        res = pd.DataFrame({"model": model_global,
                            "precision": precision_global,
                            "recall": recall_global,
                            "F-score": Fscore_global,
                            "accuracy": accuracy_global,
                            "time": time_global
                            })
        res.to_csv(path_or_buf=name_res, index=False)
    elif args.classify:
        for k in models:
            train_and_predict(models[k], k, data_type, [args.classify])
    else:
        parser.print_help()
        print()
        print("Please select mode.")




def train_and_predict(model, model_type, data_type, msg_to_classify, testData=None):
    print("\nTrain {} classifier...".format(model_type))
    t0 = time()
    model.train()
    if testData is not None:
        print("Training done, predicting...")
        preds_rdm = model.predict(testData["message"])
        t1 = time()
        print("-> Results:")
        metrics(testData["label"], preds_rdm)
        time_global.append(round(t1 - t0, 3))
        print("Execution time (sec): ", t1 - t0)
    for msg in msg_to_classify:
        print("-> Testing:  '{}'".format(msg))
        print("{}? : {}".format(data_type, bool(model.classify(process_message(msg)))))


def load_spam_data():
    filepath = "data/spam.csv"
    print("Loading {}...".format(filepath))
    # Lecture du fichier spam.csv et transformation en DataFrame Pandas.
    mails = pd.read_csv(filepath, encoding="latin-1")
    # Supression des trois dernières colonnes
    mails.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    # On renomme les colonnes v1 et v2
    mails.rename(columns={"v1": "labels", "v2": "message"}, inplace=True)
    # On imprime le nombre de spam et de non spam
    print(mails["labels"].value_counts())
    print_avg_word_count(mails)
    # On rajoute une colonne "label" avec les valeurs: spam = 1, et non spam = 0
    mails["label"] = mails["labels"].map({"ham": 0, "spam": 1})
    # On supprime la colonne "labels"
    mails.drop(["labels"], axis=1, inplace=True)
    return mails, "Spam"


def load_clickbait_data():
    filepath = "data/clickbait.csv"
    print("Loading {}...".format(filepath))
    headlines = pd.read_csv(filepath, encoding="utf-8")
    # On renomme les colonnes
    headlines.rename(columns={"headline": "message", "clickbait": "label"}, inplace=True)
    # Renvoie un sample pour diminuer la duree de traitement
    sample = headlines.loc[np.random.randint(0, headlines.shape[0], 5500)]
    sample.reset_index(inplace=True, drop=True)
    # On imprime le de nombre de clickbait et non clickbait
    print(sample["label"].value_counts())
    print_avg_word_count(sample)
    return sample, "Clickbait"


def load_rt_reviews_data():
    filepath = "data/rt_reviews.csv"
    print("Loading {}...".format(filepath))
    reviews = pd.read_csv(filepath, encoding="utf-8")
    # On renomme les colonnes
    reviews.rename(columns={"reviews": "message", "labels": "label"}, inplace=True)
    # On melange le dataframe
    sample = reviews.sample(frac=1)
    # On imprime le de nombre de reviews positives et negatives
    print(sample["label"].value_counts())
    print_avg_word_count(sample)
    return sample, "Positive review"


def print_avg_word_count(df):
    # Print the average word count of each message in a dataframe
    avg_wc = floor(np.mean([len(df["message"].get(i).split()) for i in range(len(df))]))
    print("Avg word count per message: {}".format(avg_wc))


def process_message(
        message, lower_case=True, stem=True, stop_words=True, gram=1
):
    """
    Cette fonction est très importante car c'est elle qui transforme les messages
    en une liste de mots clés essentiels: non stop et "stemmés".
    Si gram > 1 ce ne sont pas des mots clés mais des couples de mots clés qui sont
    pris en compte
    """
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [" ".join(words[i: i + gram])]
        return w
    if stop_words:
        sw = stopwords.words("english")
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


def zero_division(a, b):
    # handle division by 0 by returning 0 instead of an error
    return a / b if b else 0


class BaseClassifier:
    def __init__(self, trainData):
        self.mails, self.labels = trainData["message"], trainData["label"]

    def train(self):
        pass

    def classify(self, message):
        pass

    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]  # Nombre de messages
        self.spam_mails, self.ham_mails = (
            self.labels.value_counts()[1],
            self.labels.value_counts()[0],
        )
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        # Nombre de mots dans les spams
        self.ham_words = 0
        # Nombre de mots dans les non-spams
        self.tf_spam = dict()
        # dictionnaire avec le TF de chaque mot dans les spam
        self.tf_ham = dict()
        # dictionnaire avec le TF de chaque mot dans les non-spam
        self.idf_spam = dict()
        # dictionnaire avec le IDF de chaque mot dans les spam
        self.idf_ham = dict()

        # dicrionnaire avec le IDF de chaque mot dans les non-spam
        for i in range(noOfMessages):
            # appelle les librairies nltk
            message_processed = process_message(self.mails.get(i))
            count = list()
            # Pour sauver si un mot est apparu dans le message ou non
            # IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                    # calcule le TF d'un mot dans les spams
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                    # calcule le TF d'un mot dans les non spams
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                    # calcule le idf -> le nombre de spam qui contiennent ce mot
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1
                    # calcule le idf -> le nombre de non-spam qui contiennent ce mot

    def predict(self, testData):
        # Appelle le classifieur pour les messages du Test Set
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


class RandomClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        pass

    def classify(self, message):
        return np.random.randint(2, size=1)[0]


class TFIDFClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()

    def calc_TF_IDF(self):
        # Effectue le calcul global avec le tf_idf.
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = self.tf_spam[word] * log(
                (self.spam_mails + self.ham_mails)
                / (self.idf_spam[word] + self.idf_ham.get(word, 0))
            )
            self.sum_tf_idf_spam += self.prob_spam[word]

        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                    self.sum_tf_idf_spam + len(self.prob_spam.keys())
            )

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log(
                (self.spam_mails + self.ham_mails)
                / (self.idf_spam.get(word, 0) + self.idf_ham[word])
            )
            self.sum_tf_idf_ham += self.prob_ham[word]

        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (
                    self.sum_tf_idf_ham + len(self.prob_ham.keys())
            )

        self.prob_spam_mail, self.prob_ham_mail = (
            self.spam_mails / self.total_mails,
            self.ham_mails / self.total_mails,
        )

    def classify(self, processed_message):
        # classe les messages du test set
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.sum_tf_idf_spam + len(self.prob_spam.keys()))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.sum_tf_idf_ham + len(self.prob_ham.keys()))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam


class BowClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_prob()

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            # calcule la proba qu'un mot apparaisse dans les spams
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (
                    self.spam_words + len(self.tf_spam.keys())
            )
        for word in self.tf_ham:
            # calcule la proba qu'un mot apparaisse dans les non spams
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (
                    self.ham_words + len(self.tf_ham.keys())
            )
        self.prob_spam_mail, self.prob_ham_mail = (
            self.spam_mails / self.total_mails,
            self.ham_mails / self.total_mails,
        )

    def classify(self, processed_message):
        # classe les messages du test set
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.spam_words + len(self.prob_spam.keys()))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.ham_words + len(self.prob_ham.keys()))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam


class AbstractDTClassifier(ABC, BaseClassifier):
    # Abstract Class using a tf-idf document-term matrix for the classification
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bow = {}
        self.dt_matrix = None
        self.idf = None

    def train(self):
        # process all the messages
        processed_mails = self.mails.map(self.__process_message)

        # find word message count based on the index
        idx_cnt = np.zeros(len(self.bow))
        for v in self.bow.values(): idx_cnt[v[0]] = v[1]

        # compute the document-term matrix and finalize the training
        self.compute_dt_matrix(processed_mails, idx_cnt)

    def classify(self, processed_message):
        d_vector = self.vectorize(processed_message, np.zeros(len(self.bow)))
        return self.predict_dvector(d_vector)

    def vectorize(self, words, d_vector):
        # compute the tf-idf values of a document vector
        for word in words:
            if word in self.bow:
                d_vector[self.bow[word][0]] += 1
        word_cnt = np.sum(d_vector)
        idx = np.where(d_vector != 0)
        tf = d_vector[idx] / word_cnt
        d_vector[idx] = tf * self.idf[idx]
        return d_vector

    def __process_message(self, message):
        words = process_message(message)
        # update a bow that gives a unique index for each word
        # and the number of messages it appears
        for word in set(words):
            if word not in self.bow:
                self.bow[word] = [len(self.bow), 0]
            self.bow[word][1] += 1
        return words

    @abstractmethod
    def compute_dt_matrix(self, processed_mails, idx_cnt):
        pass

    @abstractmethod
    def predict_dvector(self, d_vector):
        pass


class SVMClassifier(AbstractDTClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svm = None

    def compute_dt_matrix(self, processed_mails, idx_cnt):
        # compute the document term matrix
        self.dt_matrix = np.zeros((processed_mails.shape[0], len(self.bow)))
        self.idf = np.log(self.dt_matrix.shape[0] / (1 + idx_cnt))

        for i in range(self.dt_matrix.shape[0]):
            super().vectorize(processed_mails.get(i), self.dt_matrix[i, :])

        self.svm = svm.SVC(C=1.0, kernel='linear', gamma='auto')
        self.svm.fit(self.dt_matrix, self.labels.to_numpy())

    def predict_dvector(self, d_vector):
        return self.svm.predict(np.reshape(d_vector, (-1, d_vector.size))) > 0


class KNNClassifier(AbstractDTClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 5

    def compute_dt_matrix(self, processed_mails, idx_cnt):
        # compute the document term matrix
        self.dt_matrix = np.zeros((processed_mails.shape[0], len(self.bow) + 2))
        self.idf = np.log(self.dt_matrix.shape[0] / (1 + idx_cnt))
        for i in range(self.dt_matrix.shape[0]):
            super().vectorize(processed_mails.get(i), self.dt_matrix[i, :-2])
            self.dt_matrix[i, -1] = self.labels[i]
            self.dt_matrix[i, -2] = norm(self.dt_matrix[i, :-2])

    def predict_dvector(self, d_vector):
        np.seterr(divide='ignore', invalid='ignore')
        d_norm = norm(d_vector)
        cosine_sim = np.dot(self.dt_matrix[:, :-2], d_vector) / (d_norm * self.dt_matrix[:, -2])
        nearest_idx = (-cosine_sim).argsort()[:self.k]
        return np.sum(self.dt_matrix[nearest_idx, -1]) >= self.k / 2


class KNNConfWeightClassifier(KNNClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxstr = {}

    def train(self):
        # process all the messages
        mails_and_labels = pd.Series(list(zip(self.mails, self.labels)))
        processed_mails = mails_and_labels.map(self.__process_message)
        # We change the function, so we can have a count for ham and a count for spam
        # find word message count based on the index
        idx_cnt = np.zeros(len(self.bow))
        for v in self.bow.values(): idx_cnt[v[0]] = v[1] + v[2]  # There are two values in bow
        # ham and spam now
        self.compute_maxstr()
        # compute the document-term matrix and finalize the training
        self.compute_dt_matrix(processed_mails, idx_cnt)

    def compute_maxstr(self):
        n = len(self.labels == 0)
        for word in self.bow:
            p_neg = (self.bow[word][1] + 1.96) / (n + 3.84)
            p_pos = (self.bow[word][2] + 1.96) / (n + 3.84)

            MinNeg = p_neg
            MaxNeg = p_neg
            MinPos = p_pos
            MaxPos = p_pos
            MinPosRelFreq = MinPos / (MinPos + MaxNeg)
            if (MinPos > MaxNeg):
                strpos = np.log2(2 * MinPosRelFreq)
            else:
                strpos = 0

            MinNegRelFreq = MinNeg / (MinNeg + MaxPos)
            if (MinNeg > MaxPos):
                strneg = np.log2(2 * MinNegRelFreq)
            else:
                strneg = 0

            self.maxstr[word] = [len(self.maxstr), max([strneg, strpos])]

    def compute_dt_matrix(self, processed_mails, idx_cnt):
        # compute the document term matrix
        self.dt_matrix = np.zeros((processed_mails.shape[0], len(self.bow) + 2))
        self.idf = np.log(self.dt_matrix.shape[0] / (1 + idx_cnt))
        for i in range(self.dt_matrix.shape[0]):
            self.vectorize(processed_mails.get(i), self.dt_matrix[i, :-2])
            self.dt_matrix[i, -1] = self.labels[i]
            self.dt_matrix[i, -2] = norm(self.dt_matrix[i, :-2])

    def vectorize(self, words, d_vector):
        maxstr_vector = np.zeros(len(d_vector))
        # compute the tf-idf values of a document vector
        for word in words:
            if word in self.bow:
                d_vector[self.bow[word][0]] += 1
                maxstr_vector[self.bow[word][0]] = self.maxstr[word][1]
                # For word present in the message, we add 1 and we give a value for maxstr
        word_cnt = np.sum(d_vector)
        idx = np.where(d_vector != 0)
        tf = d_vector[idx] / word_cnt
        d_vector[idx] = np.log(tf + 1) * maxstr_vector[idx]
        return d_vector

    def __process_message(self, message_and_label):
        message = message_and_label[0]
        label = message_and_label[1]
        words = process_message(message)
        # update a bow that gives a unique index for each word
        # and the number of messages it appears in
        for word in set(words):
            if word not in self.bow:
                self.bow[word] = [len(self.bow), 0, 0]
                # First 0 is for number of ham messages containing the word
                # Second 0 is for number of spam messages containing the word
            if label:
                self.bow[word][2] += 1
            else:
                self.bow[word][1] += 1
        return words


class SVMConfWeightClassifier(KNNConfWeightClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_dt_matrix(self, processed_mails, idx_cnt):
        # compute the document term matrix
        self.dt_matrix = np.zeros((processed_mails.shape[0], len(self.bow)))
        self.idf = np.log(self.dt_matrix.shape[0] / (1 + idx_cnt))

        for i in range(self.dt_matrix.shape[0]):
            self.vectorize(processed_mails.get(i), self.dt_matrix[i, :])

        self.svm = svm.SVC(C=1.0, kernel='linear', gamma='auto')
        self.svm.fit(self.dt_matrix, self.labels.to_numpy())

    def vectorize(self, words, d_vector):
        maxstr_vector = np.zeros(len(d_vector))
        # compute the tf-idf values of a document vector
        for word in words:
            if word in self.bow:
                d_vector[self.bow[word][0]] += 1
                maxstr_vector[self.bow[word][0]] = self.maxstr[word][1]
                # For word present in the message, we add 1 and we give a value for maxstr
        word_cnt = np.sum(d_vector)
        idx = np.where(d_vector != 0)
        tf = d_vector[idx] / word_cnt
        d_vector[idx] = tf * self.idf[idx] * maxstr_vector[idx]
        return d_vector        

    def predict_dvector(self, d_vector):
        return self.svm.predict(np.reshape(d_vector, (-1, d_vector.size))) > 0       

class GAClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_progress = False
        self.vocab = {}
        self.pos_idx = set()
        self.neg_idx = set()
        self.fittest_individual = None
        self.train_size = 0.9
        self.pop_size = 50
        self.elites = 10
        self.generations = 100
        self.mutation_rate = .001

    def train(self):
        # process all the messages
        processed_mails = self.__process_message()

        # create a training and validation set
        trn_idx = np.random.uniform(0, 1, len(processed_mails)) < self.train_size
        val_idx = np.invert(trn_idx)
        trn_set, trn_labels = processed_mails[trn_idx], self.labels.to_numpy()[trn_idx]
        val_set, val_labels = processed_mails[val_idx], self.labels.to_numpy()[val_idx]

        # create an initial population of individuals
        population = [self.__repair(np.random.rand(2, len(self.vocab)) > 0.5) for _ in range(self.pop_size)]

        # iterate over n generations
        for i in range(self.generations):
            # construct roulette wheel probabilities
            fitness = np.array([self.__get_fitness(trn_set, trn_labels, indiv) for indiv in population])
            fitness_sum = np.sum(fitness)
            rw_prob = [1/len(population) for _ in range(len(population))] if fitness_sum == 0 \
                else [fitness[i] / fitness_sum for i in range(len(fitness))]

            if self.print_progress:
                print('{} ; {}'.format(i, fitness_sum))

            # add elites
            new_pop = [population[i] for i in (-fitness).argsort()[:self.elites]]

            # generate and add offsprings
            while len(new_pop) < len(population):
                # select parents by roulette wheel
                parent_1 = population[np.random.choice(len(population), p=rw_prob)]
                parent_2 = population[np.random.choice(len(population), p=rw_prob)]

                # generate kid 1 and kid 2 by crossing over
                kid_1 = self.__crossover(parent_1, parent_2)
                kid_2 = self.__crossover(parent_1, parent_2)

                # apply some mutation
                self.__mutate(kid_1)
                self.__mutate(kid_2)

                # repair and add the offsprings to the new population
                new_pop.append(self.__repair(kid_1))
                new_pop.append(self.__repair(kid_2))

            # replace the current population with the new one
            population = new_pop

        # select the fittest individual using the validation set
        fitness = np.array([self.__get_fitness(val_set, val_labels, indiv) for indiv in population])
        fittest = population[np.argmax(fitness)]

        # eliminate redundant terms
        trn_set, trn_labels = np.hstack((trn_set, val_set)), np.hstack((trn_labels, val_labels))
        fitness = self.__get_fitness(trn_set, trn_labels, fittest)
        for i in range(fittest.shape[0]):
            for j in range(fittest.shape[1]):
                if fittest[i, j]:
                    fittest[i, j] = False
                    cur_fitness = self.__get_fitness(trn_set, trn_labels, fittest)
                    if cur_fitness >= fitness:
                        fitness = cur_fitness
                    else:
                        fittest[i, j] = True
        self.fittest_individual = fittest

        if self.print_progress:
            inv_vocab = {v: k for k, v in self.vocab.items()}
            pos_words = [inv_vocab[i] for i in range(self.fittest_individual.shape[1]) if self.fittest_individual[0, i]]
            neg_words = [inv_vocab[i] for i in range(self.fittest_individual.shape[1]) if self.fittest_individual[1, i]]
            print('Positive terms: {}'.format(pos_words))
            print('Negative terms: {}'.format(neg_words))

    def classify(self, processed_message):
        return self.__classify(processed_message, self.fittest_individual)

    def __crossover(self, parent_1, parent_2):
        # generate offspring by uniform crossing over
        mask = (np.random.rand(1, parent_1.shape[1]) > 0.5)[0]
        pos_chromosome = [parent_1[0, i] if mask[i] else parent_2[0, i] for i in range(len(mask))]
        neg_chromosome = [parent_1[1, i] if mask[i] else parent_2[1, i] for i in range(len(mask))]
        return np.array((pos_chromosome, neg_chromosome))

    def __mutate(self, individual):
        # apply random mutation to an individual
        for i in range(individual.shape[0]):
            for j in range(individual.shape[1]):
                if np.random.uniform() <= self.mutation_rate:
                    individual[i, j] = ~individual[i, j]

    def __repair(self, individual):
        # repair illegal individuals
        for i in range(individual.shape[1]):
            if individual[0, i] and individual[1, i]:
                individual[0, i] = False
            if individual[0, i] and i not in self.pos_idx:
                individual[0, i] = False
            if individual[1, i] and i not in self.neg_idx:
                individual[1, i] = False
        return individual

    def __get_fitness(self, ts, labels, individual):
        # get the fitness of an individual as the F-measure of its predictions
        # on the given training set (faster than the one used to print results)
        predictions = [self.__classify(msg, individual) for msg in ts]
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, label in enumerate(labels):
            tp += label and predictions[i]
            tn += not label and not predictions[i]
            fp += not label and predictions[i]
            fn += label and not predictions[i]
        return zero_division(tp, tp + 0.5 * (fp + fn))

    def __classify(self, processed_message, individual):
        # the occurrence of a positive term in the message requires the contextual
        # absence of the set of negative terms in order to be classified as Spam
        in_pos, in_neg = False, False
        for idx in map(lambda x: self.vocab[x], filter(lambda x: x in self.vocab, set(processed_message))):
            in_pos = True if individual[0, idx] else in_pos
            in_neg = True if individual[1, idx] else in_neg
            if in_pos and in_neg: break
        return in_pos and not in_neg

    def __process_message(self):
        processed_messages = []
        for i in range(len(self.mails)):
            words = process_message(self.mails.get(i))
            processed_messages.append(words)
            for word in set(words):
                # update global vocabulary
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                # update local vocabularies
                if self.labels[i]:
                    self.pos_idx.add(self.vocab[word])
                else:
                    self.neg_idx.add(self.vocab[word])
        return np.array(processed_messages, dtype=object)


def metrics(labels, predictions):  # Calcule les métriques
    # True Positive, True Negative, False Positive, False Negative
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels.get(i) == 1 and predictions.get(i) == 1)
        true_neg += int(labels.get(i) == 0 and predictions.get(i) == 0)
        false_pos += int(labels.get(i) == 0 and predictions.get(i) == 1)
        false_neg += int(labels.get(i) == 1 and predictions.get(i) == 0)
    precision = zero_division(true_pos, true_pos + false_pos)
    recall = zero_division(true_pos, true_pos + false_neg)
    Fscore = zero_division(2 * precision * recall, precision + recall)
    accuracy = zero_division(true_pos + true_neg, true_pos + true_neg + false_pos + false_neg)
    precision_global.append(round(precision, 3))
    recall_global.append(round(recall, 3))
    Fscore_global.append(round(Fscore, 3))
    accuracy_global.append(round(accuracy, 3))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
