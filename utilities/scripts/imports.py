import sys
import json
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import math
import random
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import multivariate_normal, pearsonr
import scipy.integrate as integrate
from sklearn.neighbors import KernelDensity
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.offline import download_plotlyjs, plot, iplot, init_notebook_mode
print('imported external packages')