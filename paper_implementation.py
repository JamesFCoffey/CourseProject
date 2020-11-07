"""Python implementation of a causal topic modeling paper.

This program implements the following paper:
    Hyun Duk Kim, Malu Castellanos, Meichun Hsu, ChengXiang Zhai, Thomas Rietz,
    and Daniel Diermeier. 2013. Mining causal topics in text data: Iterative
    topic modeling with time series feedback. In Proceedings of the 22nd ACM
    international conference on information & knowledge management (CIKM 2013).
    ACM, New York, NY, USA, 885-890. DOI=10.1145/2505515.2505612
"""

# Import libraries
import plsa
import numpy as np
import pandas as pd
import nltk

# Import data
pres_market = pd.read_csv("./data/PRES00_WTA.csv")
AAMRQ = pd.read_csv("./data/AAMRQ.csv")
AAPL = pd.read_csv("./data/AAPL.csv")

#
print(pres_market)
print(AAMRQ)
print(AAPL)
