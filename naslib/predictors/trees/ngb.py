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

def parse_params(func):
    @wraps(func)
    def wrapper(*args, identifier='base', **kwargs):
        params = dict()
        for k, v in func(*args, **kwargs).items():
            if k.startswith(identifier):
                params[k.replace(identifier, "")] = v
        return params
    return wrapper


class NGBoost(BaseTree):

    @parse_params
    def default_params(self):
        params = {
            'param:n_estimators': 505,
            'param:learning_rate': 0.08127053060223186,
            'param:minibatch_frac': 0.5081694143793387,
            'base:max_depth': 6,
            'base:max_features': 0.7920456318712875,
            'base:min_samples_leaf': 15,
            'base:min_samples_split': 20,
            #'early_stopping_rounds': 100,
            #'verbose': -1
        }
        return params

    @parse_params
    def param_func(self):
        return self.param_dict
    
    def get_random_dict(self):
        params = {
            'param:n_estimators': int(loguniform(128, 512)),
            'param:learning_rate': loguniform(.001, .1),
            'param:minibatch_frac': np.random.uniform(.1, 1),
            'base:max_depth': np.random.choice(24) + 1,
            'base:max_features': np.random.uniform(.1, 1),
            'base:min_samples_leaf': np.random.choice(18) + 2,
            'base:min_samples_split': np.random.choice(18) + 2,
            #'early_stopping_rounds': 100,
            #'verbose': -1
        }
        return params        
    
    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return (encodings, (labels-self.mean)/self.std)


    def train(self, train_data):

        # debug
        for _ in range(5):
            print('params are', self.params(identifier='param:'))
        print('end')
        # end debug

        X_train, y_train = train_data
        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **self.params(identifier='base:'))
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=True, **self.params(identifier='param:'))

        return model.fit(X_train, y_train)


    def fit(self, xtrain, ytrain, train_info=None, params=None, param_type='hpo', **kwargs):
        if param_type == 'default':
            self.params = self.default_params
        elif param_type == 'saved':
            pass
        else:
            self.params = self.run_hpo(xtrain, ytrain)
        
        return super(NGBoost, self).fit(xtrain, ytrain, params, **kwargs)

    def run_hpo(self, xtrain, ytrain, iters=1000):
        min_score = 100000
        best_params = None
        for _ in range(iters):
            self.param_dict = self.get_random_dict()
            self.params = self.param_func           
            print('trying out', self.params(identifier='base:'), self.params(identifier='param:'))
            score = self.cross_validate(xtrain, ytrain)
            if score < min_score:
                min_score = score
                best_params = self.params
                print('new best', score, best_params(identifier='base:'), best_params(identifier='param:'))
        return best_params
        
    def cross_validate(self, xtrain, ytrain):
        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **self.params(identifier='base:'))
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=True, **self.params(identifier='param:'))
        scores = cross_val_score(model, xtrain, ytrain, cv=3)
        print(scores, np.mean(scores))
        return np.mean(scores)
