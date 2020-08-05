from sklearn import linear_model as lm
import scipy
import numpy as np


class LogisticRegression(lm.LogisticRegression):

    def __init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C= C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                 l1_ratio=l1_ratio
        )

        self.cov_matrix = None


    def fit(self, X,y,sample_weight=None):
        super().fit(X,y,sample_weight=sample_weight)

        predProbs = self.predict_proba(X)

        # Design matrix -- add column of 1's at the beginning of your X_train matrix
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])

        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        V = np.diagflat(np.product(predProbs, axis=1))

        # Covariance matrix
        # Note that the @-operater does matrix multiplication in Python 3.5+, so if you're running
        # Python 3.5+, you can replace the covLogit-line below with the more readable:
        # covLogit = np.linalg.inv(X_design.T @ V @ X_design)
        self.cov_matrix = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))

        std_err = np.sqrt(np.diag(self.cov_matrix))

        self.std_err_intercept_ = std_err[0]
        self.std_err_coef_ = std_err[1:]

        self.z_intercept_ = self.intercept_/self.std_err_intercept_
        self.z_coef_ = self.coef_/self.std_err_coef_

        self.p_val_intercept_ = scipy.stats.norm.sf(abs(self.z_intercept_))*2
        self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_))*2

        return self
