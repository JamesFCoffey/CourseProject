import numpy as np
import math

def normalize(input_matrix):
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

class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]

        Update self.number_of_documents
        """
        with open(self.documents_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.split()
                if line[0] == '0' or line[0] == '1':
                    del line[0]
                self.documents.append(line)
                self.number_of_documents += 1

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

        Don't forget to normalize!
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob) # P(z | d)

        self.topic_word_prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob) # P(w | z)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob) # P(z | d)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob) # P(w | z)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")

        for i in range(self.topic_prob.shape[1]):
            self.topic_prob[:, i, :] =  np.outer(self.document_topic_prob[:, i], self.topic_word_prob[i, :])
            self.topic_prob[:, i, :] /= np.matmul(self.document_topic_prob, self.topic_word_prob)

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z) and P(z | d)
        """
        print("M step:")

        # update P(w | z)
        for i in range(number_of_topics):
            self.topic_word_prob[i, :] = np.diag(np.matmul(np.transpose(self.term_doc_matrix), self.topic_prob[:, i, :]))
            self.topic_word_prob[i, :] /= np.sum(self.topic_word_prob[i, :])

        # update P(z | d)
        for i in range(number_of_topics):
            self.document_topic_prob[:, i] = np.diag(np.matmul(self.term_doc_matrix, np.transpose(self.topic_prob[:, i, :])))
        self.document_topic_prob /= np.sum(self.document_topic_prob, axis = 1)[:, None]

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices

        Append the calculated log-likelihood to self.likelihoods
        """
        self.likelihoods.append(np.sum(np.multiply(self.term_doc_matrix, np.log(np.matmul(self.document_topic_prob, self.topic_word_prob)))))
        return self.likelihoods[-1]

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")

        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)
            previous_likelihood = current_likelihood
            current_likelihood = self.calculate_likelihood(number_of_topics)
            if np.abs(current_likelihood - previous_likelihood) < epsilon:
                break
