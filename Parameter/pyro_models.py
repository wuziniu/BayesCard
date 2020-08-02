import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
pyro.enable_validation(True)

class Pyro_dist():
    """
        A base class for all pyro distributions that will be used to build BN
    """
    def __init__(self, attr_name, condition_name=[]):
        self.attr_name = attr_name
        self.condition_name = condition_name

    def learn_parameters(self, data, model, parameters, init_loc_fn, optim, n_epoches=200, max_plate_nesting=1):
        """
            Learning parameters using variational inference
        """
        elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)

        def initialize(seed):
            global global_guide, svi
            pyro.set_rng_seed(seed)
            pyro.clear_param_store()
            global_guide = AutoDelta(poutine.block(model, expose=parameters),
                                     init_loc_fn=init_loc_fn)
            svi = SVI(model, global_guide, optim, loss=elbo)
            return svi.loss(model, global_guide, data)

        # Choose the best among 100 random initializations.
        loss, seed = min((initialize(seed), seed) for seed in range(100))
        initialize(seed)
        self.seed = seed

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        losses = []
        for i in range(n_epoches):
            loss = svi.step(data)
            losses.append(loss)

        map_estimates = global_guide(data)
        self.result = dict()
        for param in parameters:
            self.result[param] = map_estimates[param].data.numpy()
        return self.result


class GMM(Pyro_dist):
    """
        Gaussian Mixture model to model continuous data's pdf, P(X)
    """
    def __init__(self, attr_name, condition_name=[]):
        Pyro_dist.__init__(self, attr_name, condition_name)

    def learn_from_data(self, data, K=1, lr=0.01, n_epoches=200):
        if type(data) == pd.core.series.Series:
            data = torch.tensor(data.values).type(torch.FloatTensor)
        elif type(data) == np.ndarray:
            data = torch.tensor(data).type(torch.FloatTensor)

        @config_enumerate
        def model(data):
            # Global variables.
            weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
            with pyro.plate('components', K):
                scales = pyro.sample('scales', dist.LogNormal(0., 20.))
                locs = pyro.sample('locs', dist.Normal(0., 10.))

            with pyro.plate('data', len(data)):
                # Local variables.
                assignment = pyro.sample('assignment', dist.Categorical(weights))
                pyro.sample('obs', dist.Normal(locs[assignment], scales[assignment]), obs=data)

        def init_loc_fn(site):
            if site["name"] == "weights":
                # Initialize weights to uniform.
                return torch.ones(K) / K
            if site["name"] == "scales":
                return torch.tensor([(data.var() / 2).sqrt()] * K)
            if site["name"] == "locs":
                return data[torch.multinomial(torch.ones(len(data)) / len(data), K)]
            raise ValueError(site["name"])

        optim = Adam({'lr': lr, 'betas': [0.8, 0.99]})

        self.parameters = self.learn_parameters(data, model, ["weights", "scales", "locs"], init_loc_fn, optim, n_epoches)

        return self.parameters


class Categorical(Pyro_dist):
    """
        Simple categorical distribution to model continuous data's pdf, P(X)
    """

    def __init__(self, attr_name, condition_name=[]):
        Pyro_dist.__init__(self, attr_name, condition_name)


    def learn_from_data(self, data, lr=0.01, n_epoches=50):
        n_cat = data.nunique()
        data = torch.tensor(data.values).type(torch.FloatTensor)

        def model(data):
            # Global variables.
            probs = pyro.sample('probs', dist.Dirichlet(0.5 * torch.ones(n_cat)))
            with pyro.plate('data', len(data)):
                # Local variables.
                pyro.sample('obs', dist.Categorical(probs), obs=data)

        optim = pyro.optim.Adam({'lr': lr, 'betas': [0.8, 0.99]})

        def init_loc_fn(site):
            if site["name"] == "probs":
                # Initialize probs to uniform.
                return torch.ones(n_cat) / n_cat
            raise ValueError(site["name"])

        self.parameters = self.learn_parameters(data, model, ["probs"], init_loc_fn, optim,
                                                n_epoches)

        return self.parameters


class ConditionalDist(Pyro_dist):
    """
        Condition distribution to model P(X|Y1, Y2,...Yk)
    """

    def __init__(self, attr_name, condition_name=[]):
        Pyro_dist.__init__(self, attr_name, condition_name)

    def learn_from_data(self, data, lr=0.01, n_epoches=50):
        n_cat = data.nunique()
        data = torch.tensor(data.values).type(torch.FloatTensor)

        def model(data):
            # Global variables.
            probs = pyro.sample('probs', dist.Dirichlet(0.5 * torch.ones(n_cat)))
            with pyro.plate('data', len(data)):
                # Local variables.
                pyro.sample('obs', dist.Categorical(probs), obs=data)

        optim = pyro.optim.Adam({'lr': lr, 'betas': [0.8, 0.99]})

        def init_loc_fn(site):
            if site["name"] == "probs":
                # Initialize probs to uniform.
                return torch.ones(n_cat) / n_cat
            raise ValueError(site["name"])

        self.parameters = self.learn_parameters(data, model, ["probs"], init_loc_fn, optim,
                                                n_epoches)

        return self.parameters
