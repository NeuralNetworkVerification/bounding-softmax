import pickle
import numpy as np

with open("clean_NLL_Brier_mnist100.pkl", 'rb') as fp:
    scores = pickle.load(fp)
for score in scores:
    print(score)
    print(sum(scores[score]) / 100)
