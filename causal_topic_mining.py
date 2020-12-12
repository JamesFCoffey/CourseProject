"""Python implementation of a causal topic modeling paper.

This program implements the following paper:
    Hyun Duk Kim, Malu Castellanos, Meichun Hsu, ChengXiang Zhai, Thomas Rietz,
    and Daniel Diermeier. 2013. Mining causal topics in text data: Iterative
    topic modeling with time series feedback. In Proceedings of the 22nd ACM
    international conference on information & knowledge management (CIKM 2013).
    ACM, New York, NY, USA, 885-890. DOI=10.1145/2505515.2505612
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import nltk
for dependency in ("punkt", "stopwords", "brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset"):
    nltk.download(dependency)
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from normalise import normalise
import multiprocessing as mp
from tqdm import tqdm

class ITMTF(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path, time_series):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.doc_timestamps = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.lag = 5
        self.sig_cutoff = 0.95
        self.probM = 0.5
        self.min_impact_ratio = 0.1
        self.topic_prior = None
        self.mu = 0
        self.ct = []
        self.average_entropy = []
        self.average_topic_purity = []
        self.average_causality_confidence = []
        self.time_series = time_series

    def normalize(self, input_matrix):
        """
        Normalizes the rows of a 2d input_matrix so they sum to 1
        """

        row_sums = input_matrix.sum(axis=1)
        try:
            assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
        except Exception:
            raise Exception("Error while normalizing. Row(s) sum to zero")
        new_matrix = input_matrix / row_sums[:, np.newaxis]
        return new_matrix

    def token_pipeline(self, line):
        tokens = word_tokenize(line.lower())
        tokens= [x for x in tokens if x.isalnum()]
        nltk_stop_words = nltk.corpus.stopwords.words('english')
        tokens = [x for x in tokens if x not in nltk_stop_words]
        tokens = normalise(tokens, variety="AmE", verbose = False)
        tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]
        tokens = [WordNetLemmatizer().lemmatize(word, pos='n') for word in tokens]
        return tokens

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]

        Update self.number_of_documents
        """
        lines = []
        with os.scandir(self.documents_path) as it:
            for entry in it:
                with open(entry, 'r') as file:
                    while True:
                        line = file.readline()
                        if not line:
                            break
                        lines.append(line)
                        self.doc_timestamps.append(os.path.basename(entry).split(".")[0])
                        self.number_of_documents += 1
        with mp.Pool() as pool:
            self.documents = pool.map(self.token_pipeline, lines)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        word_set = set()
        for document in self.documents:
            for word in document:
                word_set.add(word)
        self.vocabulary = list(word_set)
        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document,
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        self.term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))
        for i, document in enumerate(self.documents):
            for j, word in enumerate(self.vocabulary):
                self.term_doc_matrix[i][j] = document.count(word)

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob
        """
        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = self.normalize(self.document_topic_prob) # P(z | d)

        self.topic_word_prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = self.normalize(self.topic_word_prob) # P(w | z)

        self.topic_prior = np.zeros(self.topic_word_prob.shape)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = self.normalize(self.document_topic_prob) # P(z | d)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = self.normalize(self.topic_word_prob) # P(w | z)

        self.topic_prior = np.zeros(self.topic_word_prob.shape)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        for i in range(self.topic_prob.shape[1]):
            self.topic_prob[:, i, :] =  np.outer(self.document_topic_prob[:, i], self.topic_word_prob[i, :])
            self.topic_prob[:, i, :] /= np.matmul(self.document_topic_prob, self.topic_word_prob)

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z) and P(z | d)
        """
        # update P(w | z)
        for i in range(number_of_topics):
            numerator = np.diag(np.matmul(np.transpose(self.term_doc_matrix), self.topic_prob[:, i, :]))
            denominator = np.sum(numerator)
            self.topic_word_prob[i, :] = (self.mu * self.topic_prior[i, :] + numerator) / (self.mu + denominator)

        # update P(z | d)
        for i in range(number_of_topics):
            self.document_topic_prob[:, i] = np.diag(np.matmul(self.term_doc_matrix,
                                                               np.transpose(self.topic_prob[:, i, :])))
        self.document_topic_prob /= np.sum(self.document_topic_prob, axis = 1)[:, None]

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices

        Append the calculated log-likelihood to self.likelihoods
        """
        self.likelihoods.append(np.sum(np.multiply(self.term_doc_matrix,
                                                   np.log(np.matmul(self.document_topic_prob, self.topic_word_prob)))))
        return self.likelihoods[-1]

    def process(self, number_of_topics, max_plsa_iter, epsilon, mu, itmtf_iter):

        """
        Model topics.
        """
        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in tqdm(range(max_plsa_iter), desc="Initial PLSA"):
            self.expectation_step()
            self.maximization_step(number_of_topics)
            previous_likelihood = current_likelihood
            current_likelihood = self.calculate_likelihood(number_of_topics)
            if np.abs(current_likelihood - previous_likelihood) < epsilon:
                break
        for iteration in tqdm(range(itmtf_iter), desc="ITMTF Loop"):
            self.mu = mu
            self.topic_level_causality()
            wc = self.word_level_causality()
            self.generate_topic_prior(wc)
            for iteration in tqdm(range(max_plsa_iter), desc="PLSA at end of ITMTF iter."):
                self.expectation_step()
                self.maximization_step(number_of_topics)
                previous_likelihood = current_likelihood
                current_likelihood = self.calculate_likelihood(number_of_topics)
                if np.abs(current_likelihood - previous_likelihood) < epsilon:
                    break
            self.metrics()

    def impact_value(self, data, lag):
        model = VAR(data)
        results = model.fit(lag)
        numerator = 0
        for i in range(1, lag + 1):
            numerator += results.params[
                results.params.columns.values[0]]['L' + str(i) + '.' + results.params.columns.values[1]]
        return numerator / np.abs(lag)

    def build_TS(self):
        ts_data = np.zeros((len(pd.unique(pd.Series(self.doc_timestamps))), self.document_topic_prob.shape[1]))
        for k, timestamp in enumerate(pd.unique(pd.Series(self.doc_timestamps))):
            doc_i = pd.Series(self.doc_timestamps)[pd.Series(self.doc_timestamps) == timestamp].index.values
            ts_data[k] = np.sum(self.document_topic_prob[doc_i], axis=0)
        ts = pd.DataFrame(ts_data, index=pd.to_datetime(pd.unique(pd.Series(self.doc_timestamps))))
        return ts

    def topic_level_causality(self):
        ts = self.build_TS()
        self.ct = []
        for topic_i in range(self.document_topic_prob.shape[1]):
            text_series = ts[topic_i]
            text_series = text_series.rolling(3, center=True, min_periods=2).mean()
            text_series = text_series.diff()[1:]
            input_df = pd.concat([text_series, self.time_series], axis=1, join="inner").sort_index()
            gc_res = grangercausalitytests(input_df, maxlag=self.lag, verbose=False)
            sig_cxt = []
            for i in range(1, len(gc_res) + 1):
                sig_cxt.append(1 - gc_res[i][0]['params_ftest'][1])
            if np.max(sig_cxt) > self.sig_cutoff:
                self.ct.append((topic_i, sig_cxt.index(np.max(sig_cxt)) + 1))
        return

    def top_words(self, topic):
        cumul_prob = 0
        tw = []
        tw_i = 0
        twp_series = pd.Series(self.topic_word_prob[topic,:]).sort_values(ascending=False)
        while True:
            if cumul_prob + twp_series.iloc[tw_i] > self.probM:
                break
            else:
                cumul_prob += twp_series.iloc[tw_i]
                tw.append(twp_series.index[tw_i])
                tw_i += 1
        tw.sort()
        return tw

    def build_WS(self, tw):
        ws = pd.DataFrame(self.term_doc_matrix[:,tw], columns=tw,
                          index=pd.to_datetime(self.doc_timestamps)).groupby(level=0).sum()
        return ws

    def word_level_causality(self):
        topic_list = []
        word_list = []
        iv_list = []
        sig_list = []
        for causal_topic, causal_lag in self.ct:
            tw = self.top_words(causal_topic)
            ws = self.build_WS(tw)
            for word in tw:
                word_series = ws[word]
                word_series = word_series.rolling(3, center=True, min_periods=2).mean()
                word_series = word_series.diff()[1:]
                input_df = pd.concat([word_series, self.time_series], axis=1, join="inner").sort_index()
                gc_res = grangercausalitytests(input_df, maxlag=self.lag, verbose=False)
                sig_cxw = []
                for i in range(1, len(gc_res) + 1):
                    sig_cxw.append(1 - gc_res[i][0]['params_ftest'][1])
                if np.max(sig_cxw) > self.sig_cutoff:
                    topic_list.append(causal_topic)
                    word_list.append(word)
                    iv_list.append(self.impact_value(input_df, sig_cxw.index(np.max(sig_cxw)) + 1))
                    sig_list.append(np.max(sig_cxw))
        return pd.DataFrame({"Topic": topic_list, "Word": word_list, "Impact_Value": iv_list, "Significance": sig_list})

    def generate_topic_prior(self, wc):
        self.topic_prior = np.zeros(self.topic_word_prob.shape)
        topic_iter = 0
        for topic in wc["Topic"].unique():
            pos_rows = len(wc[wc["Topic"] == topic][wc["Impact_Value"] >= 0])
            neg_rows = len(wc[wc["Topic"] == topic][wc["Impact_Value"] < 0])
            if pos_rows / neg_rows < self.min_impact_ratio:
                wc = wc.drop(wc[wc["Topic"] == topic][wc["Impact_Value"] >= 0].index)
                wc.loc[wc["Topic"] == topic, "Significance"] -= self.sig_cutoff
                wc.loc[wc["Topic"] == topic, "Significance"] /= \
                wc.loc[wc["Topic"] == topic, "Significance"].sum()
                self.topic_prior[topic_iter, wc.loc[wc["Topic"] == topic, "Word"]] = \
                wc.loc[wc["Topic"] == topic, "Significance"]
                topic_iter += 1
                if topic_iter >= self.topic_word_prob.shape[0]:
                    break
            elif neg_rows / pos_rows < self.min_impact_ratio:
                wc = wc.drop(wc[wc["Topic"] == topic][wc["Impact_Value"] < 0].index)
                wc.loc[wc["Topic"] == topic, "Significance"] -= self.sig_cutoff
                wc.loc[wc["Topic"] == topic, "Significance"] /= \
                wc.loc[wc["Topic"] == topic, "Significance"].sum()
                self.topic_prior[topic_iter, wc.loc[wc["Topic"] == topic, "Word"]] = \
                wc.loc[wc["Topic"] == topic, "Significance"]
                topic_iter += 1
                if topic_iter >= self.topic_word_prob.shape[0]:
                    break
            else:
                wc.loc[(wc["Impact_Value"] >= 0) & (wc["Topic"] == topic), "Significance"] -= self.sig_cutoff
                wc.loc[(wc["Impact_Value"] >= 0) & (wc["Topic"] == topic), "Significance"] /= \
                wc.loc[(wc["Impact_Value"] >= 0) & (wc["Topic"] == topic), "Significance"].sum()
                self.topic_prior[topic_iter, wc.loc[(wc["Impact_Value"] >= 0) & (wc["Topic"] == topic), "Word"]] = \
                wc.loc[(wc["Impact_Value"] >= 0) & (wc["Topic"] == topic), "Significance"]
                topic_iter += 1
                if topic_iter >= self.topic_word_prob.shape[0]:
                    break
                wc.loc[(wc["Impact_Value"] < 0) & (wc["Topic"] == topic), "Significance"] -= self.sig_cutoff
                wc.loc[(wc["Impact_Value"] < 0) & (wc["Topic"] == topic), "Significance"] /= \
                wc.loc[(wc["Impact_Value"] < 0) & (wc["Topic"] == topic), "Significance"].sum()
                self.topic_prior[topic_iter, wc.loc[(wc["Impact_Value"] < 0) & (wc["Topic"] == topic), "Word"]] = \
                wc.loc[(wc["Impact_Value"] < 0) & (wc["Topic"] == topic), "Significance"]
                topic_iter += 1
                if topic_iter >= self.topic_word_prob.shape[0]:
                    break
        return

    def create_eval_df(self):
        topic_list = []
        word_list = []
        iv_list = []
        for topic_index in range(self.topic_word_prob.shape[0]):
            tw = self.top_words(topic_index)
            ws = self.build_WS(tw)
            for word in tw:
                word_series = ws[word]
                word_series = word_series.rolling(3, center=True, min_periods=2).mean()
                word_series = word_series.diff()[1:]
                input_df = pd.concat([word_series, self.time_series], axis=1, join="inner").sort_index()
                gc_res = grangercausalitytests(input_df, maxlag=self.lag, verbose=False)
                sig_cxw = []
                for i in range(1, len(gc_res) + 1):
                    sig_cxw.append(1 - gc_res[i][0]['params_ftest'][1])
                if np.max(sig_cxw) > self.sig_cutoff:
                    topic_list.append(topic_index)
                    word_list.append(word)
                    iv_list.append(self.impact_value(input_df, sig_cxw.index(np.max(sig_cxw)) + 1))
        return pd.DataFrame({"Topic": topic_list, "Word": word_list, "Impact_Value": iv_list})

    def metrics(self):
        wc = self.create_eval_df()
        entropies = []
        topic_purities = []
        for topic in wc["Topic"].unique():
            pos_rows = len(wc[wc["Topic"] == topic][wc["Impact_Value"] >= 0])
            neg_rows = len(wc[wc["Topic"] == topic][wc["Impact_Value"] < 0])
            p_prob = pos_rows / (pos_rows + neg_rows)
            n_prob = neg_rows / (pos_rows + neg_rows)
            entropies.append(p_prob * np.log(p_prob) + n_prob * np.log(n_prob))
            topic_purities.append(100 + 100 * (p_prob * np.log(p_prob) + n_prob * np.log(n_prob)))
        self.average_entropy.append(np.mean(entropies))
        self.average_topic_purity.append(np.mean(topic_purities))
        ts = self.build_TS()
        sig_cxt_maximums = []
        for topic_i in range(self.document_topic_prob.shape[1]):
            text_series = ts[topic_i]
            text_series = text_series.rolling(3, center=True, min_periods=2).mean()
            text_series = text_series.diff()[1:]
            input_df = pd.concat([text_series, self.time_series], axis=1, join="inner").sort_index()
            gc_res = grangercausalitytests(input_df, maxlag=self.lag, verbose=False)
            sig_cxt = []
            for i in range(1, len(gc_res) + 1):
                sig_cxt.append(1 - gc_res[i][0]['params_ftest'][1])
            sig_cxt_maximums.append(np.max(sig_cxt))
        self.average_causality_confidence.append(np.mean(sig_cxt_maximums))
        return
