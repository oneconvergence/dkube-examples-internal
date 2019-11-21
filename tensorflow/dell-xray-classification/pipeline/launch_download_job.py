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


# def update_sync_flag(status):
#     if (status == "true") or (force_run_all_stages == "true"):
#         flag = "true"
#     else:
#         flag = "false"
#     if status == "error":
#         flag = "false"
#     print("run_next_stages={}".format(flag))
#     with open(OUTPUT_FILE, 'w') as out:
#         out.write(flag)


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


# def get_filename(cd):
#     """
#     Get filename from content-disposition
#     """
#     if not cd:
#         return None
#     fname = re.findall('filename=(.+)', cd)
#     if len(fname) == 0:
#         return None
#     return fname[0]


# def generate_sync_flag(folder_name):
#     # update_sync_flag("true")
#     # return
#     log_folder = "/tmp/train-logs"
#     sync_flag = None
#     try:
#         if folder_name.endswith("tar"):
#             my_tar = tarfile.open(folder_name)
#             my_tar.extractall(path=log_folder)
#             my_tar.close()
#         else:
#             with zipfile.ZipFile(folder_name, 'r') as zip_ref:
#                 zip_ref.extractall(log_folder)
#     except Exception as err:
#         print(err)
#     for filename in os.listdir(log_folder):
#         if filename.endswith("log"):
#             filename = "{}/{}".format(log_folder, filename)
#         else:
#             filename = "{}/{}".format(log_folder, filename)
#             filename += "/{}".format(os.listdir(filename)[0])
#         try:
#             for line in reversed(open(filename).readlines()):
#                 if "DATASET SYNCED=" in line:
#                     sync_flag = line.split("=")[-1].strip()
#                     break
#             if sync_flag == "true":
#                 update_sync_flag("true")
#             else:
#                 update_sync_flag("false")
#         except Exception as err:
#             print(err)
#             update_sync_flag("error")


# def check_ds_synced(url, user, token, job_name):
#     download_url = Template('$url/dkube/v2/ext/users/$user/class/datajob/'
#                             'jobs/$jobname/logdownload')
#     header = {"content-type": "application/keyauth.api.v1+json",
#               'Authorization': 'Bearer {}'.format(token)}
#     if url[-1] == '/':
#         url = url[:-1]
#     try:
#         url = download_url.substitute({'url': url,
#                                        'user': user,
#                                        'jobname': job_name})
#         download_header = header.copy()
#         download_header['content-type'] = 'application/octet-stream'
#         session = requests.Session()
#         resp = session.get(url, headers=download_header, verify=False)
#         if resp.status_code != 200:
#             print('Unable to download the logs for the job %s' % jobname)
#             return 'ERROR'
#         file = get_filename(resp.headers.get('content-disposition'))
#         if file:
#             open(file, 'wb').write(resp.content)
#             generate_sync_flag(file)
#             job_id = file.split('.')[0]
#             return job_id
#         else:
#             print("Could't get filename")
#             return None
#     except Exception as e:
#         return None

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
                                        "target": "chexnet"
                                    }}}
            data = json.dumps(data)
            resp = session.post(
                url, data=data, headers=create_header, verify=False)
            if resp.status_code != 200:
                print('Unable to start job %s' % job_name)
                return None
        except Exception as e:
            print("Error: ", e)
            return None
    start_job(url, user, token, ws_name, ds_name, JOB_NAME)
    poll_for_job_completion(url,user,token,JOB_NAME)



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--user', type=str)
#     parser.add_argument('--auth_token', type=str)
#     parser.add_argument('--access_url', type=str)
#     parser.add_argument('--ds_name', type=str)
#     parser.add_argument('--ws_name', type=str)
#     ar = parser.parse_args()
#     print("launching the resources")
#     job_name = JOB_NAME
#     ds_name = json.loads(ar.ds_name)[0]
#     start_job(ar.access_url, ar.user, ar.auth_token, ar.ws_name, ds_name, job_name)
#     time.sleep(5)
#     status = poll_for_job_completion(ar.access_url, ar.user, ar.auth_token, job_name)
#     # if status:
#         # wait for couple fo secs to store logs for datajob
#         # immediate parsing of logs fails sometime as logs are not available immediately
#     delete_generated_dataset(ar.access_url, ar.user, ar.auth_token, TARGET_DATASET)


# if __name__ == "__main__":
#     main()
