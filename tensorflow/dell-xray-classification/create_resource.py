import os
import argparse
import json
import requests
import datetime
import time
from string import Template
from requests.packages import urllib3

WS_SOURCE_LINK = "https://github.com/oneconvergence/dkube-examples/tree/dell-model-1.4.1-pipeline/tensorflow/dell-xray-classification"
DATASET_URL = "https://github.com/oneconvergence/dkube-examples/tree/dell-model-1.4.1-pipeline/tensorflow/dell-xray-classification/dataset"
WS_NAME = "chexnet-ws"
DS_NAME = "chexnet"

def create_ws(url, user, token):
    create_url = Template('$url/dkube/v2/users/$user/datums')
    header = {"content-type": "application/keyauth.api.v1+json",
              'Authorization': 'Bearer {}'.format(token)}
    if url[-1] == '/':
        url = url[:-1]
    try:
        url = create_url.substitute({'url': url,
                                     'user': user})
        create_header = header.copy()
        session = requests.Session()
        data = {"class": "program",
                "name": WS_NAME,
                "remote": False,
                "source": "git",
                "tags": [],
                "url": WS_SOURCE_LINK}
        data = json.dumps(data)
        resp = session.post(
            url, data=data, headers=create_header, verify=False)
        if resp.status_code != 200:
            print('Unable to create workspace %s, It may be already exist' % WS_NAME)
            return None
    except Exception as e:
        return None

def create_ds(url, user, token):
    create_url = Template('$url/dkube/v2/users/$user/datums')
    header = {"content-type": "application/keyauth.api.v1+json",
              'Authorization': 'Bearer {}'.format(token)}
    if url[-1] == '/':
        url = url[:-1]
    try:
        url = create_url.substitute({'url': url,
                                     'user': user})
        create_header = header.copy()
        session = requests.Session()
        data = {"class": "dataset",
                "name": DS_NAME,
                "remote": False,
                "source": "git",
                "tags": [],
                "url": DATASET_URL}
        data = json.dumps(data)
        resp = session.post(
            url, data=data, headers=create_header, verify=False)
        if resp.status_code != 200:
            print('Unable to create dataset %s, It may be already exist' % DS_NAME)
            return None
    except Exception as e:
        return None

def poll_for_resource_creation(url, user, token, class_name, name):
    poll_count = 500
    sleep_time = 5  # sec
    created = False
    get_url = Template('$url/dkube/v2/users/$user/datums/class/$class/?shared=false')
    header = {"content-type": "application/keyauth.api.v1+json",
              'Authorization': 'Bearer {}'.format(token)}
    try:
        url = get_url.substitute({'url': url,
                                  'user': user,
                                  'class': class_name})
        get_header = header.copy()
        session = requests.Session()
        for i in range(poll_count):
            resp = session.get(
                url, headers=get_header, verify=False)
            if resp.status_code != 200:
                print('Unable to get info for %s' % name)
                return None
            data = resp.json()
            print("polling for {} to be in ready state".format(class_name))
            for dataset in data['data'][0]['datums']:
                if dataset['name'] == name:
                    if dataset['generated']['status']['state'] == 'ready':
                        created = True
                    break
            if created:
                break
            time.sleep(sleep_time)
        if created:
            print("{} is in ready state".format(class_name))
        else:
            print("Timeout: {} is not in ready state".format(class_name))
    except Exception as e:
        print("Error while polling for {} to be in ready state".format(class_name))
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str)
    parser.add_argument('--access_url', type=str)
    parser.add_argument('--auth_token', type=str)
    ar = parser.parse_args()
    if ar.user:
        user = ar.user
    if ar.access_url:
        url = ar.access_url
    if ar.auth_token:
        token = ar.auth_token

    create_ws(url, user, token)
    poll_for_resource_creation(url, user, token, "program", WS_NAME)
    create_ds(url, user, token)
    poll_for_resource_creation(url, user, token, "dataset", DS_NAME)
