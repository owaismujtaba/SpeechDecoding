import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from src.components.utils.utils import standardise
from scipy.stats import pearsonr
from src.components.utils.utils import compute_mfcc_similarity_metrics
import config as config
import pdb

class Trainer:
    def __init__(self, subject, mfcc, eeg):
        self.subject = subject
        self.mfcc = mfcc
        self.eeg = eeg
        self.n_folds = 5

    def train(self, model, model_name):
        self.model_name = model_name
        kf = KFold(self.n_folds, shuffle=False)

        rec_mfcc = np.zeros(self.mfcc.shape)
        results = np.zeros((self.n_folds, 2))
        self.eeg = standardise(self.eeg)

        for k, (train, test) in enumerate(kf.split(self.mfcc)):
            X_train = self.eeg[train]
            y_train = self.mfcc[train]
            X_test = self.eeg[test]
            y_test = self.mfcc[test]

            model.train(X_train, y_train)

            rec_mfcc[test] = model.model.predict(X_test)

            results[k] = self.evaluate_model(y_test, rec_mfcc[test])

        
        print(f'sub-{self.subject} has mean correlation of  {np.mean(results)}')
        self.save_results(rec_mfcc, results)

    def evaluate_model(self, actual, predictions):
        cosine_similarity, mcd = compute_mfcc_similarity_metrics(actual, predictions)
        return np.array([cosine_similarity, mcd])
    
    def save_results(self, rec_mfcc, results):
        res_dir = Path(config.cur_dir, 'results', self.model_name, f'sub-{self.subject}')
        os.makedirs(res_dir, exist_ok=True)

        np.save(Path(res_dir, 'rec_mfcc.npy'),rec_mfcc)
        np.save(Path(res_dir, 'orig_mfcc.npy'),self.mfcc)
        np.save(Path(res_dir, 'results.npy'),results)