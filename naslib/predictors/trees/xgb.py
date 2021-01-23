import xgboost as xgb

from naslib.predictors.trees import BaseTree
import numpy as np
from naslib.predictors.trees.ngb import loguniform

class XGBoost(BaseTree):

    @property
    def default_hyperparams(self):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': "rmse",
            # 'early_stopping_rounds': 100,
            'booster': 'gbtree',
            # NOTE: if using these hyperparameters XGB predicts the same
            # values always on NB201
            # 'max_depth': 13,
            # 'min_child_weight': 39,
            # 'colsample_bytree': 0.2545374925231651,
            # 'learning_rate': 0.008237525103357958,
            # 'alpha': 0.24167936088332426,
            # 'lambda': 31.393252465064943,
            # 'colsample_bylevel': 0.6909224923784677,
            # 'verbose': -1
        }
        return params


    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return xgb.DMatrix(encodings)
        else:
            return xgb.DMatrix(encodings, label=((labels-self.mean)/self.std))


    def train(self, train_data):
        #NOTE: in nb301 num_boost_round=20000
        return xgb.train(self.hyperparams, train_data, num_boost_round=100)

    def predict(self, data):
        return self.model.predict(self.get_dataset(data))

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        return super(XGBoost, self).fit(xtrain, ytrain, params, **kwargs)

    def get_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': "rmse",
                #'early_stopping_rounds': 100,
                'booster': 'gbtree',
                #NOTE: if using these hyperparameters XGB predicts the same
                # values always on NB201
                'max_depth': np.random.choice(range(1,15)),
                'min_child_weight': np.random.choice(range(1,100)),
                'colsample_bytree': np.random.uniform(.0, 1.0),
                'learning_rate': loguniform(.001, .1),
                #'alpha': 0.24167936088332426,
                #'lambda': 31.393252465064943,
                'colsample_bylevel': np.random.uniform(.0, 1.0),
                #'verbose': -1
            }
        return params