# -*- coding: utf-8 -*-
"""modelinitialisierung.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VtelAhzX3F9E3IpkIZ-AcNjs4lbYyCbe
"""

import torch
from torch import distributions
import datetime
import numpy as np
from scipy.stats import poisson, nbinom

# Details Model Parameter
dict_model_param = {'lower': {'R0': 0.001, 'phi': 0, 'sigma': 0.001, 'alpha': 0.001},
                    'upper': {'R0': 3, 'phi': 50, 'sigma': 0.5, 'alpha': 0.05},
                    'value': {'R0': 1.5, 'phi': 25, 'sigma': 0.1, 'alpha': 0.028},
                    'scale': {'R0': 1, 'phi': 10, 'sigma': 0.02, 'alpha': 0.002}}

def get_dict_model_param():
    return dict_model_param

# Transformation
def bij_transform(prime, lower, upper):
  """Recieves a prime value of type tensor in [-inf, inf] and returns value in [lower, upper]"""
  bij = 1 / (1 + torch.exp(-prime / upper))
  scale = upper - lower
  return scale * bij + lower

def bij_transform_inv(transf, lower, upper):
  """Inverse transformation - Recieves a value of type tensor in [lower, upper] and returns value in [-inf, inf]"""
  return -torch.log(((upper - lower) / (transf - lower) - 1) ** upper)

def transform_prime_variables(dict_param):
  for key in dict_param['real_values'].keys():
    dict_param['real_values'][key] = bij_transform(dict_param['prime_values'][key], dict_model_param['lower'][key], dict_model_param['upper'][key])
  return dict_param
  

# Initialize Model Parameter
def initialize_prime_param(param, device, dtype):
  value = dict_model_param['value'][param]
  lower = dict_model_param['lower'][param]
  upper = dict_model_param['upper'][param]
  prime = bij_transform_inv(torch.tensor(value, device=device, dtype=dtype), lower, upper).detach().clone().requires_grad_(True)
  return prime

def initialize_parameter(parameter, device, dtype):
  dict_parameter = {'prime_values':{}, 'real_values': {}}
  for param in parameter:
    dict_parameter['prime_values'][param] = initialize_prime_param(param, device, dtype)
    dict_parameter['real_values'][param] = bij_transform(dict_parameter['prime_values'][param], dict_model_param['lower'][param], dict_model_param['upper'][param])
  return dict_parameter

def initialize_epsilon(num_observations, sigma, device, dtype):
  epsilon_t = torch.zeros(num_observations, device=device, dtype=dtype)
  epsilon_t[0] = torch.distributions.Normal(torch.tensor(0., requires_grad=False, device=device, dtype=dtype), sigma.detach()).rsample()
  for t in range(1, num_observations):
      epsilon_t[t] = torch.distributions.Normal(epsilon_t[t - 1].detach(), sigma.detach()).rsample()
  return epsilon_t.requires_grad_(True)

# Observations
def initialize_observations(df_observations, start='2020-02-26', end='2022-01-31', observations=['number_of_deaths', 'newly_infected', 'hospitalization'], rolling_avg=1):
    for observation in observations:
        df_observations[observation] = df_observations[observation].rolling(rolling_avg).mean()
    
    # filter observations
    time_period = (df_observations['Date'] >= start) & (df_observations['Date'] < end)
    columns = ['Date'] + observations
    df_obs_filtered = df_observations.loc[time_period][columns].reset_index(drop=True)
    
    # calc initial newly infected
    time_format = "%Y-%m-%d"
    dt_start = datetime.datetime.strptime(start, time_format)
    if (dt_start < datetime.datetime.strptime('2020-03-10', time_format)):
      initial_newly_infected = np.arange(1, 6)
    else:
      initial_start = (dt_start - datetime.timedelta(6)).strftime(time_format)
      initial_time_period = (df_observations['Date'] >= initial_start) & (df_observations['Date'] < start)
      initial_newly_infected = df_observations.loc[initial_time_period]['newly_infected'].to_numpy()
      
    return df_obs_filtered, initial_newly_infected

# Loss Functions

def calc_random_walk_loss(epsilon_t, sigma, device, dtype):
  """Takes epsilon_t and sigma as an input and returns the random walk loss."""
  days = len(epsilon_t)
  loc = epsilon_t[:days-1]
  scale = sigma * torch.ones(days - 1, device=device, dtype=dtype)
  mvn = distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=torch.diag(scale))
  ll = mvn.log_prob(epsilon_t[1:days])
  return -ll

def calc_mse(expected, observed):
  diff = expected - observed
  square = diff.square()
  msr = square.mean()
  return msr

def calc_poisson_loss(expected, observed):
    sum = 0
    for i, val in enumerate(expected):
        mu = val
        k = observed[i]
        sum += poisson.pmf(k, mu)
    return sum/(i+1) #TODO: ask Andreas what should we do with values at the end and parameters of negative bin dist

def calc_negative_binomnial_loss(expected, observed, phi):
    sum = 0
    for i, val in enumerate(expected):
        mu = val
        k = observed[i]
        sum += nbinom.pmf(k, mu, phi)
    return sum/(i + 1)  # TODO: ask Andreas what should we do with values at the end and parameters of negative bin dist


def calc_prior_loss(dict_param, device, dtype):
  """Takes the dictionary of parameter as an input and calculates the prior loss.
  The prior loss is calculated by using the log-probability."""

  ll = torch.tensor(0.0, device=device, dtype=dtype)

  for key in dict_param['real_values'].keys():
    value = torch.tensor(dict_model_param['value'][key], device=device, dtype=dtype)
    scale = torch.tensor(dict_model_param['scale'][key], device=device, dtype=dtype)
    parameter = dict_param['real_values'][key]
    ll += distributions.normal.Normal(loc=value, scale=scale).log_prob(parameter)

  return -ll
