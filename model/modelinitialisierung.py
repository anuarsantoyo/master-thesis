# -*- coding: utf-8 -*-
"""modelinitialisierung.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VtelAhzX3F9E3IpkIZ-AcNjs4lbYyCbe
"""

import torch
from torch import distributions

def initialize_observations(df_observations, start='2020-02-26', end='2022-01-31', observations=['Number_of_deaths', 'Confirmed_cases', 'Admissions_hospital']):
  time_period = (df_observations['Date'] >= start) & (df_observations['Date'] < end)
  columns = ['Date'] + observations
  df_obs_filtered = df_observations.loc[time_period][columns].reset_index(drop=True)
  return df_obs_filtered

def initialize_epsilon(num_observations, sigma, device, dtype):
  epsilon_t = torch.zeros(num_observations, device=device, dtype=dtype)
  epsilon_t[0] = torch.distributions.Normal(torch.tensor(0., requires_grad=False, device=device, dtype=dtype), sigma.detach()).rsample()
  for t in range(1, num_observations):
      epsilon_t[t] = torch.distributions.Normal(epsilon_t[t - 1].detach(), sigma.detach()).rsample()
  return epsilon_t.requires_grad_(True)