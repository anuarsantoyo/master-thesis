from model.modelhelper import *
from torch import distributions

class RandomWalk():
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
