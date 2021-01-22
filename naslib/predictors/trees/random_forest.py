from sklearn.ensemble import RandomForestRegressor as RF

from naslib.predictors.trees import BaseTree
import numpy as np

class RandomForestPredictor(BaseTree):

    @property
    def default_hyperparams(self):
        #NOTE: Copied from NB301
        params = {
            'n_estimators': 116,
            'max_features': 0.17055852159745608,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'bootstrap': False,
            #'verbose': -1
        }
        return params


    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return (encodings, (labels-self.mean)/self.std)

    def train(self, train_data):
        X_train, y_train = train_data
        model = RF(**self.hyperparams)
        return model.fit(X_train, y_train)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams
        return super(RandomForestPredictor, self).fit(xtrain, ytrain, params, **kwargs)

    def get_random_hyperparams(self):
        params = {
            'n_estimators': np.random.choice(range(80,120)),
            'max_features': 0.17055852159745608,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'bootstrap': False,
            # 'verbose': -1
        }
        return params