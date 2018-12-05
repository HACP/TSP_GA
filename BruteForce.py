import pandas as pd
import numpy as np
import pylab as plt
import time
import itertools
from sklearn.neighbors import KDTree

np.random.seed(2018)

data = pd.read_csv("../data/all/cities.csv")

def primesfrom2to(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in xrange(int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)/3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))/3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

data['prime'] = data['CityId'].isin(primesfrom2to(data.shape[0]))

def path_length(data,city_sequence):
    if len(np.unique(city_sequence)) < data.shape[0]:
        print "Not enough cities"
        return
    data_paths = pd.concat([data,data[data['CityId']==0]])
    #data_paths['Step'] = np.arange(data.shape[0]+1)
    data_paths_iteration= pd.DataFrame({'CityId': city_sequence,'Step':np.arange(data.shape[0]+1)})
    data_paths_iteration = pd.merge(data_paths_iteration,data_paths,on='CityId',how='left').drop_duplicates()
    data_diff = data_paths_iteration[['X','Y']].diff()
    data_diff['CityId'] = data_paths_iteration['CityId']
    data_diff['prime'] = data_paths_iteration['prime']*1.
    data_diff['prime_shift'] = data_diff['prime'].shift(1)
    data_diff['step10'] = ((data_paths_iteration['Step'])%10==0)*1.
    data_diff['length'] = np.sqrt(data_diff['X']**2+data_diff['Y']**2)
    d0 =  data_diff['length'].dropna().sum()/1.e6
    d1 =  0.1*data_diff[(data_diff['step10']==1)&(data_diff['prime_shift']==0)]['length'].sum()/1.e6
    return d0 + d1

path0 = data['CityId'].tolist() + [0]
print path_length(data,path0)

data_bulk = data[data['CityId']>0]

path1 = [0] + data_bulk.sort_values(['X','Y'])['CityId'].tolist() + [0]
print path_length(data,path1)

N_L = 200
data_bulk['nX'] = np.floor((data_bulk['X']/(data_bulk['X'].max()*1.000001))*N_L)
data_bulk['nY'] = np.floor((data_bulk['Y']/(data_bulk['Y'].max()*1.000001))*N_L)

path1 = [0] + data_bulk.sort_values(['nX','nY','X','Y'])['CityId'].tolist() + [0]
print path_length(data,path1)

t0 = time.time()
seq = [0]
data_kNN = data.copy()
ii = 0
while len(seq) < data.shape[0]:
    index =  data_kNN[data_kNN['CityId']==seq[-1]].index[0]
    #print index
    kdt = KDTree(data_kNN[['X','Y']], leaf_size=30, metric='euclidean')
    index_nn = kdt.query(np.array(data[['X','Y']])[index:index+1], k=2, return_distance=False)[0][1]
    seq.append(index_nn)
    data_kNN.drop(index=index)

    if ii % 1000 == 0:
        print ii, time.time() - t0
    ii = ii + 1
print seq

path2_length = path_length(data,seq+[0])

pd.DataFrame({'Path':seq + [0]}).to_csv('/files/submission_' + str(path2_length) + '.csv',index=False)
