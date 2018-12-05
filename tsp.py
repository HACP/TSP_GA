import numpy as np
import pandas as pd

import tsp_utils

np.random.seed(2018)

data = pd.read_csv("../data/cities.csv")

data['prime'] = data['CityId'].isin(tsp_utils.primesfrom2to(data.shape[0]))

print data.head()
