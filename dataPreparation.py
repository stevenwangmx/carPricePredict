import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

%matplotlib inline

auto_prices = pd.read_csv('Automobile price data _Raw_.csv')
auto_prices.head(20)
