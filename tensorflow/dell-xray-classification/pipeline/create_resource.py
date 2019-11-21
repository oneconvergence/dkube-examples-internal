'''
import os
import argparse
import json
import requests
import datetime
import time
from string import Template
from requests.packages import urllib3






def create_ws(url, user, token, ws_name, ws_link):
    # WS_SOURCE_LINK = "https://github.com/oneconvergence/dkube-examples/tree/dell-model-1.4.1-pipeline/tensorflow/dell-xray-classification"
    WS_SOURCE_LINK = ws_link
    # WS_NAME = "chexnet-ws"
    WS_NAME = ws_name
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

def create_ds(url, user, token, ds_name, ds_link):
    # DATASET_URL = "https://github.com/oneconvergence/dkube-examples/tree/dell-model-1.4.1-pipeline/tensorflow/dell-xray-classification/dataset"
    DATASET_URL = ds_link
    # DS_NAME = "chexnet"
    DS_NAME = ds_name
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

def create_resource(user, access_url, auth_token, ws_name, ws_link, ds_name, ds_link):
    print("Inside create_resource")
    import os
    import argparse
    import json
    import requests
    import datetime
    import time
    from string import Template
    from requests.packages import urllib3
    print('After import')
    print("user {} \n access_url {} \n auth_token {} \n".format(user,access_url,auth_token))
    create_ws(url, user, token, ws_name, ws_link)
    poll_for_resource_creation(url, user, token, "program", ws_name)
    create_ds(url, user, token, ds_name, ds_link)
    poll_for_resource_creation(url, user, token, "dataset", ds_name)
'''
def create_resource_job(url, user, token, ws_name, ws_link, ds_name, ds_link):
    print("[create_resource_job] init")
    import os
    def install(package):
        command = "pip install "+package
        os.system(command)
    install('requests')
    def create_ws(url, user, token, ws_name, ws_link):
        print("[create_ws] init")
        from string import Template
        import requests
        import json
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        print("[create_ws] module imported")
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        create_url = Template('$url/dkube/v2/users/$user/datums')
        header = {"content-type": "application/keyauth.api.v1+json",
                  'Authorization': 'Bearer {}'.format(token)}
        if url[-1] == '/':
            url = url[:-1]
        try:
            print("[create_ws] try block")
            url = create_url.substitute({'url': url,
                                         'user': user})
            create_header = header.copy()
            session = requests.Session()
            data = {"class": "program",
                    "gitaccess": {
                        "private": False},
                    "name": ws_name,
                    "remote": False,
                    "source": "git",
                    "tags": [],
                    "url": ws_link}
            data = json.dumps(data)
            print("[create_ws] before request")
            resp = session.post(
                url, data=data, headers=create_header, verify=False)
            print("[create_ws] after request: {}".format(resp))
            if resp.status_code != 200:
                print('Unable to create workspace %s, It may be already exist' % ws_name)
                return None
            print("workspace {} added".format(ws_name))
        except Exception as e:
            return None
    
    
    
    def create_ds(url, user, token, ds_name, ds_link):
        print("[create_ds] init")
        from string import Template
        import requests
        import json
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        print("[create_ds] module imported")
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        create_url = Template('$url/dkube/v2/users/$user/datums')
        header = {"content-type": "application/keyauth.api.v1+json",
                  'Authorization': 'Bearer {}'.format(token)}
        if url[-1] == '/':
            url = url[:-1]
        try:
            print("[create_ds] try block")
            url = create_url.substitute({'url': url,
                                         'user': user})
            create_header = header.copy()
            session = requests.Session()
            data = {"class": "dataset",
                    "gitaccess": {
                        "private": False},
                    "name": ds_name,
                    "remote": False,
                    "source": "git",
                    "tags": [],
                    "url": ds_link}
            data = json.dumps(data)
            print("[create_ds] before request")
            resp = session.post(
                url, data=data, headers=create_header, verify=False)
            print("[create_ds] after request: {}".format(resp))
            if resp.status_code != 200:
                print('Unable to create dataset %s, It may be already exist' % ds_name)
                return None
            print("dataset {} added".format(ds_name))
        except Exception as e:
            return None
    
    
    def poll_for_resource_creation(url, user, token, class_name, name):
        print("[poll_resource] init")
        import time
        from string import Template
        import requests
        import json
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        print("[poll_resource] module imported")
        poll_count = 500
        sleep_time = 5  # sec
        created = False
        get_url = Template('$url/dkube/v2/users/$user/datums/class/$class/?shared=false')
        header = {"content-type": "application/keyauth.api.v1+json",
                  'Authorization': 'Bearer {}'.format(token)}
        try:
            print("[poll_resource] try block")
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
    create_ws(url, user, token, ws_name, ws_link)
    print("[create_resource_job] after ws")
    poll_for_resource_creation(url, user, token, "program", ws_name)
    # create_ds(url, user, token, ds_name, ds_link)
    # print("[create_resource_job] after ds")
    # poll_for_resource_creation(url, user, token, "dataset", ds_name)
    with open('output.txt','w') as out_file:
        out_file.write("create resource job end")
    print("[create_resource_job] finish")
