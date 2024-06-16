#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd


# In[4]:


YEAR = 2023
MONTH = 3


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{YEAR:04d}-{MONTH:02d}.parquet')


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ### Q1. What's the standard deviation of the predicted duration for this dataset?

# In[9]:


y_pred.std()


# ### Q2. Preparing the output
# First, let's create an artificial ride_id column:

# In[10]:


df['ride_id'] = f'{YEAR:04d}/{MONTH:02d}_' + df.index.astype('str')


# Next, write the ride id and the predictions to a dataframe with results.

# In[14]:


df_predictions = df[['ride_id']].copy()
df_predictions['predictions'] = y_pred
df_predictions


# 
# Save it as parquet:

# In[16]:


output_file = f'predictions_{YEAR:04d}-{MONTH:02d}.parquet'
output_file


# In[17]:


df_predictions.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[18]:


get_ipython().system('ls -lh')

