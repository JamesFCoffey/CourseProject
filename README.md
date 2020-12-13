# CS 410: Text Information Systems Course Project
# Documentation

## Overview of the function of the code
This code implements the following paper:
> Hyun Duk Kim, Malu Castellanos, Meichun Hsu, ChengXiang Zhai, Thomas Rietz, and Daniel Diermeier. 2013. Mining causal topics in text data: Iterative topic modeling with time series feedback. In Proceedings of the 22nd ACM international conference on information & knowledge management (CIKM 2013). ACM, New York, NY, USA, 885-890. DOI=10.1145/2505515.2505612

The code takes times series data and a corpus of text documents both with time stamps from the same period as input and outputs a set of causal topics with the lag that correlates them to the time series data.

It does this through the iterative algorithm given below:
1. Apply probabilistic latent semantic analysis (PLSA) to the document corpus to generate a preselected number (input parameter) of topics topics.
2. Use the Granger causality test correlate topics with the time series data and output correlations with a significance (one minus p-value) above a cutoff value (0.95 default) to a list of (topic, lag) tuples.
3. For each topic in the list of (topic, lag) tuples, use the Granger causality test to find causal words with above signifcance above the cutoff among top words. By default the top words are the those with the highest probability words whose summed probability does not surpass 50%.
4. Separate positive and negative impact words of each topic into individual topics and ignore the lesser topic if it is below a ratio of the major topic (by default, less than 0.1).
5. With the newly separated topics from step 4, define prior word probabilities according the significances found in step 3.
6. Apply PLSA to the document corpus using the prior word probabilities incorporated into the M-step of the EM algoritm.
7. Repeat 2-6 until a spectified number of iterations is performed.

This code can be used to text mining as to find causal topics correlated with external time series data. For example, it can be used to find topics correlated with the movement of stock prices. It can also be used to find topics correlated with the outcome of an election.

## Implementation
The code is implemented in an Python class called `ITMTF` in a module called `causal_topic_mining.py`. A portion of the code was adapted from CS 410 MP3. The class is composed of the following:

### Class Variables
- `self.documents`: The text document corpus.
- `self.doc_timestamps`: The timestamps of the documents in the corpus.
- `self.vocabulary`: The vocabulary.
- `self.likelihoods`: The log likelihoods.
- `self.documents_path`: The path to the text document corpus.
- `self.term_doc_matrix`: The term document matrix.
- `self.document_topic_prob`: P(z | d)
- `self.topic_word_prob`: P(w | z)
- `self.topic_prob`: P(z | d, w)
- `self.number_of_documents`: The number of documents.
- `self.vocabulary_size`: The size of the vocabulary.
- `self.lag` (default,  `5`): The maximum lag to which evaluate Granger causality.
- `self.sig_cutoff` (default,  `0.95`): The significance cutoff.
- `self.probM` (default,  `0.5`): The maximum sum of the probabilities of the top words for a topic.
- `self.min_impact_ratio` (default,  `0.1`): The ratio below which to discard minority impact words in a topic.
- `self.topic_prior`: The prior word probabilities.
- `self.mu`: The strength of the prior in each iteration.
- `self.ct`: The list of (topic, lag) tuples for causal topics discovered by the algorithm.
- `self.average_entropy`: The average entropy of the topics for each iteration of the algorithm.
- `self.average_topic_purity`: The average topic purity of the topics for each iteration of the algorithm.
- `self.average_causality_confidence`: The average causality confidence of the topics for each iteration of the algorithm.
- `self.time_series`: The time series data saved on intialization of the class.

### Class Functions
- `normalize(input_matrix)`: Normalizes rows two-dimensional matrix to sum to one.
- `token_pipeline(line)`: Tokenizes line of input text.
- `build_corpus()`: Reads documents from the corpus, tokenizes them, and stores them. It also stores the timestamps of the documents and records the number of documents.
- `build_vocabulary()`: Constructs the vocabulary and records the size of the vocabulary.
- `build_term_doc_matrix()`:Constructs the term document matrix.
- `initialize_randomly(number_of_topics)`: Randomly initializes the probability distributions for P(z | d) and P(w | z). It also intializes the topic prior.
- `initialize_uniformly(number_of_topics)`:  Uniformly initializes the probability distributions for P(z | d) and P(w | z). It also intializes the topic prior.
- `initialize(number_of_topics, random=False)`: Calls an intialization function.
- `expectation_step()`: The E-step of the EM algorithm for PLSA where it updates P(z | w, d).
- `maximization_step(number_of_topics)`: The of the EM algorithm for PLSA where it M-step updates P(w | z) and P(z | d).
- `process(number_of_topics, max_plsa_iter, epsilon, mu, itmtf_iter)`: The master control loop for the iterative topic modeling with time series feedback algorithm.
- `impact_value(data, lag)`: Calculates the impact value.
- `build_TS()`: Builds a topic stream.
- `topic_level_causality()`: Finds the causal topics from the topic streams and time series data.
- `top_words(topic)`: Finds the top words in a topic.
- `build_WS(tw)`: Builds a word stream.
- `word_level_causality()`: Finds the significant words and their impact value for each topic using the word streams and time series data.
- `generate_topic_prior(wc)`: Generates the topic prior.
- `create_eval_df()`: Creates a dataframe from which to evaluate metrics.
- `metrics()`: Evaluates the average entropy, average topic purity, and average causality confidence of the topics.

## Usage of the software
### Installation
Download the module `causal_topic_mining.py` to your project folder.

The module uses `Python 3.7` and requires the following libraries:
- `numpy`
- `pandas`
- `statsmodels`
- `nltk`
- `normalise`
- `tqdm`

`demonstration_notebook.ipynb` requires the additional library:
- `matplotlib`

### Demonstration
A demonstration of the use of the class `ITMTF` is given in the Jupyter notebook `demonstration_notebook.ipynb`.

#### Import the module.
`from causal_topic_mining import ITMTF`

#### Create an object from ITMTF class
`itmtf = ITMTF(documents_path, time_series)`

The path is to the text corpus of text files named YYYY-MM-DD.txt (YYYY = 4 digit year, MM = 2 digit month, DD = 2 digit day) and each line in the text file is a separate document. The time series is a Pandas series where the indicies are the dates in pd.dateime format and the time series is a stationary series.

#### Tokenize the text corpus and record document timestamps
`itmtf.build_corpus()`

#### Build vocabulary
`itmtf.build_vocabulary()`

#### Run ITMTF algorithm
`itmtf.process(number_of_topics = 30, max_plsa_iter = 1, epsilon = 0.001, mu = 1000, itmtf_iter = 5)`

## Team Contributions
- James Coffey (Captain)
  - Wrote `Proposal.pdf`
  - Wrote `Progress Report.pdf`
  - Wrote `causal_topic_mining.py`
  - Wrote `demonstration_notebook.ipynb`
  - Wrote `README.md` documentation
  - Gave tutorial presentation
- Praveen Bhushan
  - Wrote tutorial presentation
  - Gave tutorial presentation
