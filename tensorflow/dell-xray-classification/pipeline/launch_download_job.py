import os
import argparse
import re
import requests
import time
import zipfile
import tarfile
import datetime
import json
from string import Template
from requests.packages import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_FILE = "/tmp/run_next_stages"
TARGET_DATASET = "chexnet-preprocessed"

JOB_NAME = "chexnet-data-download-job-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

def delete_generated_dataset(url, user, token, ds_name):
    ds_delete_url = Template('$url/dkube/v2/users/$user/datums/class/dataset')
    header = {"content-type": "application/keyauth.api.v1+json",
              'Authorization': 'Bearer {}'.format(token)}
    if url[-1] == '/':
        url = url[:-1]
    try:
        url = ds_delete_url.substitute({'url': url,
                                       'user': user})
        delete_header = header.copy()
        session = requests.Session()
        data = json.dumps({'datums': [ds_name]})
        resp = session.delete(url, data=data, headers=delete_header, verify=False)
        if resp.status_code != 200:
            print('Unable to delete Dataset %s' % ds_name)
            return None
    except Exception as e:
        return None

def download_job(url,user,token,ws_name,ds_name):
    import os
    def install(package):
        command = "pip install "+package
        os.system(command)
    install('requests')
    import re
    import requests
    import time
    # import zipfile
    # import tarfile
    import datetime
    import json
    from string import Template
    from requests.packages import urllib3
    JOB_NAME = "chexnet-data-download-job-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    print("After import")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    def poll_for_job_completion(url, user, token, name):
        poll_count = 500
        sleep_time = 5  # sec
        completed = False
        error = False
        get_url = Template('$url/dkube/v2/users/$user/jobs/class/datajob/?shared=false')
        header = {"content-type": "application/keyauth.api.v1+json",
                  'Authorization': 'Bearer {}'.format(token)}
        try:
            poll_url = get_url.substitute({'url': url,
                                           'user': user})
            get_header = header.copy()
            session = requests.Session()
            for i in range(poll_count):
                resp = session.get(
                    poll_url, headers=get_header, verify=False)
                if resp.status_code != 200:
                    print('Unable to get info for %s' % name)
                    return None
                data = resp.json()
                print("polling for completion of datajob {}".format(name))
                for dataset in data['data'][0]['jobs']:
                    if dataset['name'] == name:
                        if dataset['parameters']['generated']['status']['state'] == 'COMPLETE':
                            completed = True
                            break
                        if dataset['parameters']['generated']['status']['state'] == 'ERROR':
                            error = True
                            break
                if completed or error:
                    break
                time.sleep(sleep_time)
            if error:
                print("Custom datajob {} for dataset sync is in error state".format(
                    name))
                return False
            if completed:
                print("Custom datajob {} for dataset sync is completed".format(
                    name))
                return True
            else:
                print("Unable to complete custom job for datset sync")
                return False
        except Exception as e:
            print("Error while running custom job for dataset sync")
            return False


    def start_job(access_url, user, token, ws_name, ds_name, job_name):
        create_url = Template('$url/dkube/v2/users/$user/jobs')
        header = {"content-type": "application/keyauth.api.v1+json",
                  'Authorization': 'Bearer {}'.format(token)}
        if access_url[-1] == '/':
            access_url = access_url[:-1]
        poll_flag =True
        # $url/dkube/v2/users/$user/datums/class/dataset/datum/chexnet-download-ds
        check_url = Template('$url/dkube/v2/users/$user/datums/class/dataset/datum/chexnet-preprocessed')
        check_url = check_url.substitute({'url': access_url,
                                         'user': user})
        create_header = header.copy()
        print("Before request for dataset check")
        print("check_url : {}".format(check_url))
        resp = requests.get(check_url, headers=create_header, verify=False)
        print("response: {}".format(resp.json()))
        if resp.status_code==200:
            print("chexnet-download-ds dataset already exist, skipping dataset download")
            poll_flag = False
            print("poll_flag: {}".format(poll_flag))
            return  poll_flag

        try:
            url = create_url.substitute({'url': access_url,
                                         'user': user})
            create_header = header.copy()
            session = requests.Session()    
            data = {"name": job_name,
                    "parameters": {"class": "datajob",
                                   "datajob": {
                                        "executor": {"choice": "custom",
                                                     "custom": {"image": {"path": "docker.io/ocdr/dkube-datascience-tf-cpu:v1.14"}}},
                                        "workspace": {"program": "{}:{}".format(user, ws_name),
                                                      "script": 'sudo -E python3 download_NIH_dataset.py --user=\"{}\" --auth_token=\"{}\" --access_url=\"{}\"'.format(user, token, access_url),
                                                      "gitcommit": {}},
                                        "datasets": ["{}:{}".format(user, ds_name)],
                                        "kind": "preprocessing",
                                        "target": "chexnet-download-ds" ##mountpath
                                    }}}
            data = json.dumps(data)
            resp = session.post(
                url, data=data, headers=create_header, verify=False)
            if resp.status_code != 200:
                print('Unable to start job %s' % job_name)
                return False
            return True
        except Exception as e:
            print("Error: ", e)
            return False

    poll_flag=start_job(url, user, token, ws_name, ds_name, JOB_NAME)
    print("poll_flag before if ")
    if poll_flag == True:
        poll_for_job_completion(url,user,token,JOB_NAME)
    else:
        print("poll_flag is False")
        return None