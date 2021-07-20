import logging

import numpy as np
import copy
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from naslib.utils import generate_kfold, cross_validation
from naslib.benchmarks.predictors.predictor_config_space import PredictorConfigSpace

logger = logging.getLogger(__name__)


def get_config(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    return cfg


class SMACRunner:

    def __init__(self, xtrain, ytrain, predictor_type, predictor, metric='kendalltau'):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.predictor = predictor
        self.predictor_type = predictor_type
        self.metric = metric

        logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    def predictor_run(self, config):
        logger.info(f'Starting cross validation')
        n_train = len(self.xtrain)
        split_indices = generate_kfold(n_train, 3)

        predictor = copy.deepcopy(self.predictor)

        hyperparams = get_config(config)
        predictor.set_hyperparams(hyperparams)

        print('Hyperparams: ', hyperparams)
        print('----------------------')

        cv_score = cross_validation(self.xtrain, self.ytrain, predictor, split_indices, self.metric)
        print('Cross Validation score: ', cv_score)
        logger.info(f'Finished')

        return 1 - cv_score

    def run(self):
        # Scenario object
        config = PredictorConfigSpace(self.predictor_type)
        cs = config.build_config_space()
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                             "wallclock-limit": 7200,
                             # max. number of function evaluations; for this example set to a low number
                             "cs": cs,  # configuration space
                             "deterministic": "true",
                             "limit_resources": False
                             })
        smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                        tae_runner=self.predictor_run)
        def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(), instance='1', seed=0)[1]
        print("Default Value: %.2f" % (def_value))

        # Optimize, using a SMAC-object
        print("Optimizing! Depending on your machine, this might take a few minutes.")

        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        inc_value = smac.get_tae_runner().run(config=incumbent, instance='1', seed=0)[1]
        print("Optimized Value: %.2f" % inc_value)
        return get_config(incumbent)
