from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace


class PredictorConfigSpace:
    def __init__(self, predictor_type):

        self.predictor_type = predictor_type.upper()

    def build_config_space(self):
        cs = ConfigurationSpace()
        # Conditions
        if self.predictor_type == 'XGB':
            # XGB
            max_depth = UniformIntegerHyperparameter("max_depth", 1, 15, default_value=6)
            min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
            colsample_bytree = UniformFloatHyperparameter('colsample_bytree', 0.0, 1.0, default_value=1.0)
            colsample_bylevel = UniformFloatHyperparameter('colsample_bylevel', 0.0, 1.0, default_value=1.0)
            learning_rate = UniformFloatHyperparameter('learning_rate', 0.0001, 1.0, default_value=0.1, log=True)
            cs.add_hyperparameters([max_depth, min_child_weight, colsample_bytree, colsample_bylevel, learning_rate])
        elif self.predictor_type == 'BLR':
            # BLR
            alpha = UniformFloatHyperparameter('alpha', 1e-5, 1e5, default_value=1.0)
            beta = UniformIntegerHyperparameter('beta', 1e-5, 1e5, default_value=100)
            basis_func = CategoricalHyperparameter('basis_func', ['linear_basis_func', 'quadratic_basis_func'],
                                                   default_value='linear_basis_func')
            do_mcmc_blr = CategoricalHyperparameter('do_mcmc', choices=[True, False])
            n_hypers = UniformIntegerHyperparameter('n_hypers', 1, 50, default_value=20)
            chain_length = UniformIntegerHyperparameter('chain_length', 50, 500, default_value=100)
            burnin_steps = UniformIntegerHyperparameter('burnin_steps', 50, 500, default_value=100)
            cs.add_hyperparameters([alpha, beta, basis_func, do_mcmc_blr, n_hypers, chain_length, burnin_steps])
        elif self.predictor_type == 'BANANAS':
            # BANANAS
            num_layers = UniformIntegerHyperparameter('num_layers', 5, 25, default_value=20)
            layer_width = UniformIntegerHyperparameter('layer_width', 5, 25, default_value=20)
            regularization = UniformFloatHyperparameter('regularization', 0, 1, default_value=0.2)
            batch_size = UniformIntegerHyperparameter('batch_size', 32, 256, default_value=32)
            lr = UniformFloatHyperparameter('lr', 0.00001, 0.1, log=True)
            cs.add_hyperparameters([num_layers, layer_width, regularization, batch_size, lr])
        elif self.predictor_type == 'BOHAMIANN':
            # BOHAMIANN
            num_steps = UniformIntegerHyperparameter('num_steps', default_value=100)
            keep_every = UniformIntegerHyperparameter('keep_every', default_value=5)
            lr_bohamiann = UniformFloatHyperparameter('lr', 0.00001, 0.1, log=True)
            num_burn_in_steps = UniformIntegerHyperparameter('num_burn_in_steps', default_value=10)
            cs.add_hyperparameters([num_steps, keep_every, lr_bohamiann, num_burn_in_steps])
        elif self.predictor_type == 'BONAS':
            # BONAS
            gcn_hidden = UniformIntegerHyperparameter('gcn_hidden', 16, 128, default_value=64, log=True)
            batch_size_bonas = UniformIntegerHyperparameter('batch_size', 32, 256, default_value=128, log=True)
            lr_bonas = UniformFloatHyperparameter('lr', 0.00001, 0.1, log=True)
            cs.add_hyperparameters([gcn_hidden, batch_size_bonas, lr_bonas])
        elif self.predictor_type == 'DNGO':
            batch_size_dngo = UniformIntegerHyperparameter('batch_size', 32, 256, default_value=128, log=True)
            num_epochs = UniformIntegerHyperparameter('num_epochs', 100, 1000, default_value=500)
            lr_dngo = UniformFloatHyperparameter('lr', 0.00001, 0.1, default_value=0.01, log=True)
            n_units_1 = UniformIntegerHyperparameter('n_units_1', 10, 100, default_value=50)
            n_units_2 = UniformIntegerHyperparameter('n_units_2', 10, 100, default_value=50)
            n_units_3 = UniformIntegerHyperparameter('n_units_3', 10, 100, default_value=50)
            alpha_dngo = UniformFloatHyperparameter('alpha', 1e-5, 1e5, default_value=1.0)
            beta_dngo = UniformIntegerHyperparameter('beta', 1e-5, 1e5, default_value=100)
            do_mcmc_dngo = CategoricalHyperparameter('do_mcmc', [True, False], default_value=False)
            n_hypers_dngo = UniformIntegerHyperparameter('n_hypers', 1, 50, default_value=20)
            chain_length_dngo = UniformIntegerHyperparameter('chain_length', 1000, 4000, default_value=1000)
            burnin_steps_dngo = UniformIntegerHyperparameter('burnin_steps', 1000, 4000, default_value=1000)
            cs.add_hyperparameters(
                [batch_size_dngo, num_epochs, lr_dngo, n_units_1, n_units_2, n_units_3, alpha_dngo, beta_dngo,
                 do_mcmc_dngo, n_hypers_dngo, chain_length_dngo, burnin_steps_dngo])
        elif self.predictor_type == 'LGB':
            # LGB
            num_leaves = UniformIntegerHyperparameter('num_leaves', 10, 100, default_value=31)
            lr_lgb = UniformFloatHyperparameter('lr', 0.00001, 0.1, default_value=0.5, log=True)
            feature_fraction = UniformFloatHyperparameter('feature_fraction', 0.1, 1, default_value=0.9)
            cs.add_hyperparameters([num_leaves, lr_lgb, feature_fraction])
        elif self.predictor_type == 'GCN':
            # GCN
            gcn_hidden_gcn = UniformIntegerHyperparameter('gcn_hidden', 64, 200, default_value=64, log=True)
            batch_size_gcn = UniformIntegerHyperparameter('batch_size', 5, 32, default_value=7, log=True)
            lr_gcn = UniformFloatHyperparameter('lr', 0.1, 0.00001, default_value=0.0001, log=True)
            wd = UniformFloatHyperparameter('wd', 0.1, 0.00001, default_value=3e-4, log=True)
            cs.add_hyperparameters([gcn_hidden_gcn, batch_size_gcn, lr_gcn, wd])
        elif self.predictor_type == 'RF':
            n_estimators = UniformIntegerHyperparameter('n_estimators', 16, 128, default_value=116, log=True)
            max_features = UniformFloatHyperparameter('max_features', 0.1, 0.9, default_value=0.17055852159745608,
                                                      log=True)
            min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, default_value=2)
            min_samples_split = UniformIntegerHyperparameter('min_samples_split', 2, 20, default_value=2)
            cs.add_hyperparameters([n_estimators, max_features, min_samples_leaf, min_samples_split])
        elif self.predictor_type == 'GP':
            # GP
            kernel_gp = CategoricalHyperparameter('kernel', ['RBF', 'Matern32', 'Matern52'], default_value='RBF')
            lengthscale_gp = UniformFloatHyperparameter('lengthscale', 1e-5, 1e5, default_value=10)
            cs.add_hyperparameters([kernel_gp, lengthscale_gp])
        elif self.predictor_type == 'SparseGP':
            # Sparse GP
            kernel_s_gp = CategoricalHyperparameter('kernel', ['RBF', 'Matern32', 'Matern52'], default_value='RBF')
            lengthscale_s_gp = UniformFloatHyperparameter('lengthscale', 1e-5, 1e5, default_value=10)
            cs.add_hyperparameters([kernel_s_gp, lengthscale_s_gp])
        elif self.predictor_type == 'VarSparseGP':
            # Var Sparse GP
            kernel_vs_gp = CategoricalHyperparameter('kernel', ['RBF', 'Matern32', 'Matern52'],
                                                     default_value='RBF')
            lengthscale_vs_gp = UniformFloatHyperparameter('lengthscale', 1e-5, 1e5, default_value=10)
            cs.add_hyperparameters([kernel_vs_gp, lengthscale_vs_gp])

        return cs
