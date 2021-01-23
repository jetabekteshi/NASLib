from functools import wraps
import numpy as np

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from naslib.predictors.trees import BaseTree

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

# def parse_params(func):
#     @wraps(func)
#     def wrapper(*args, identifier='base', **kwargs):
#         params = dict()
#         for k, v in func(*args, **kwargs).items():
#             if k.startswith(identifier):
#                 params[k.replace(identifier, "")] = v
#         return params
#     return wrapper


class NGBoost(BaseTree):

    @property
    def default_hyperparams(self):
        params = {
            'param': {
                'n_estimators': 505,
                'learning_rate': 0.08127053060223186,
                'minibatch_frac': 0.5081694143793387},
            'base': {
                'max_depth': 6,
                'max_features': 0.7920456318712875,
                'min_samples_leaf': 15,
                'min_samples_split': 20}
                #'early_stopping_rounds': 100,
            #'verbose': -1
        }
        return params

    # @parse_params
    # def param_func(self):
    #     return self.hyperparams
    
    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return (encodings, (labels-self.mean)/self.std)


    def train(self, train_data):

        X_train, y_train = train_data
        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **self.hyperparams['base'])
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=True, **self.hyperparams['param'])

        return model.fit(X_train, y_train)


    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
            # self.params = self.param_func
        
        return super(NGBoost, self).fit(xtrain, ytrain, params, **kwargs)

    def get_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
                'param': {
                    'n_estimators': int(loguniform(128, 512)),
                    'learning_rate': loguniform(.001, .1),
                    'minibatch_frac': np.random.uniform(.1, 1)},
                'base':{
                    'max_depth': np.random.choice(24) + 1,
                    'max_features': np.random.uniform(.1, 1),
                    'min_samples_leaf': np.random.choice(18) + 2,
                    'min_samples_split': np.random.choice(18) + 2}
                #'early_stopping_rounds': 100,
                #'verbose': -1
            }
        return params
