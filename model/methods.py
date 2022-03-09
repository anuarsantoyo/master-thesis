from model.modelhelper import *
from torch import distributions

class RandomWalk:
    def __init__(self, n_observations, device, dtype):
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations

        self.dict_model_param =get_dict_model_param()
        self.sigma_prime = initialize_prime_param('sigma', self.device, self.dtype)
        self.sigma = bij_transform(self.sigma_prime, self.dict_model_param['lower']['sigma'],
                                   self.dict_model_param['upper']['sigma'])

        self.R0_prime = initialize_prime_param('R0', self.device, self.dtype)
        self.R0 = bij_transform(self.R0_prime, self.dict_model_param['lower']['R0'], self.dict_model_param['upper']['R0'])

        self.epsilon_t = initialize_epsilon(self.n_observations, self.sigma, self.device, self.dtype)

    def get_parameters(self):
        return [self.epsilon_t, self.sigma_prime, self.R0_prime]

    def calculate_R(self):
        self.R0 = bij_transform(self.R0_prime, self.dict_model_param['lower']['R0'], self.dict_model_param['upper']['R0'])

        self.sigma = bij_transform(self.sigma_prime, self.dict_model_param['lower']['sigma'], self.dict_model_param['upper']['sigma'])

        # Initialize eta
        eta = torch.zeros(self.n_observations, device=self.device, dtype=self.dtype)  # transformed reproduction number
        # calculate Rt: the basic reproduction number
        # basic reproduction number as a latent random walk
        beta_0 = torch.log(self.R0)
        eta[0] = beta_0
        eta[1:] =beta_0 + self.epsilon_t[0:self.n_observations-1]
        R = torch.exp(eta)
        return R

    def calculate_loss(self):
        return self.calc_random_walk_loss() + self.calc_prior_loss()

    def calc_random_walk_loss(self):
      loc = self.epsilon_t[:self.n_observations-1]
      scale = self.sigma * torch.ones(self.n_observations - 1, device=self.device, dtype=self.dtype)
      mvn = distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=torch.diag(scale))
      ll = mvn.log_prob(self.epsilon_t[1:self.n_observations])
      return -ll

    def calc_prior_loss(self):
        ll = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        value = torch.tensor(self.dict_model_param['value']['sigma'], device=self.device, dtype=self.dtype)
        scale = torch.tensor(self.dict_model_param['scale']['sigma'], device=self.device, dtype=self.dtype)

        ll += distributions.normal.Normal(loc=value, scale=scale).log_prob(self.sigma)

        value = torch.tensor(self.dict_model_param['value']['R0'], device=self.device, dtype=self.dtype)
        scale = torch.tensor(self.dict_model_param['scale']['R0'], device=self.device, dtype=self.dtype)
        ll += distributions.normal.Normal(loc=value, scale=scale).log_prob(self.R0)

        return -ll

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1= nn.Linear(1,6)
        self.fc2 = nn.Linear(6,10)
        self.fc3 = nn.Linear(10,4)
        self.fc4 = nn.Linear(4,1)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return 5*torch.sigmoid(x)

model = Net()

class TwoClusterNN:
    def __init__(self, n_observations=None, device=None, dtype=None):
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations
        self.model = Net()

    def get_parameters(self):
        return list(set(self.model.parameters()))

    def calculate_R(self,x):
        R = self.model(x)
        return R

    def calculate_loss(self):
        return None