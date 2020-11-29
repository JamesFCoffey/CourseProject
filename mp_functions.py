import nltk
for dependency in ("punkt", "stopwords", "brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset"):
    nltk.download(dependency)
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from normalise import normalise

def token_pipeline(line):
    tokens = word_tokenize(line.lower())
    tokens= [x for x in tokens if x.isalnum()]
    nltk_stop_words = nltk.corpus.stopwords.words('english')
    tokens = [x for x in tokens if x not in nltk_stop_words]
    tokens = normalise(tokens, variety="AmE", verbose = False)
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]
    tokens = [WordNetLemmatizer().lemmatize(word, pos='n') for word in tokens]
    return tokens
