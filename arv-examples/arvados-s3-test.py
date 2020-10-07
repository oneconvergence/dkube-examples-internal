#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import json
import os


# In[2]:


DATA_DIR = '/opt/dkube/input'
config = json.load(open(os.path.join(DATA_DIR,'config.json')))


# In[3]:


with open(os.path.join(DATA_DIR,'credentials'), 'r') as f:
    creds = f.read()


# In[4]:


access_key = creds.split('\n')[1].split('=')[-1].strip()
secret_key = creds.split('\n')[2].split('=')[-1].strip()


# In[5]:


session = boto3.session.Session()
s3_client = boto3.resource(
    service_name='s3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    endpoint_url=config['Endpoint']
)


# In[6]:


my_bucket = s3_client.Bucket(config['Bucket'])


# In[7]:


for file in my_bucket.objects.all():
    print(file.key)


# In[8]:


my_bucket.download_file(Filename = 'Data0000.dat',Key = 'CMU-1/Data0000.dat')

