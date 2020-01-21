'''Create resource job: This method contains three inner method responsible for 
1. create workspace
2. create dataset
3. poll for resource creation

inputs:
- url: url for which DKube UI is available
- user: active user in DKube UI
- token: authorized token (developer setting token)
- ws_link: workspace link for adding workspace to DKube 
- ds_name: Dataset name to be provided to the given dataset in DKube 
- ds_link: dataset link for adding dataset to DKube 
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
            # data = {"class": "program",
            #         "gitaccess": {
            #             "private": False},
            #         "name": ws_name,
            #         "remote": False,
            #         "source": "git",
            #         "tags": [],
            #         "url": ws_link}
            data  = {"class":"program",
            "gitaccess":{"branch":"dell-model-1.5",
            "credentials":{},
            "url":ws_link},
            "name":ws_name,
            "source":"git",
            "tags":[]}
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
            # data = {"class": "dataset",
            #         "gitaccess": {
            #             "private": False},
            #         "name": ds_name,
            #         "remote": False,
            #         "source": "git",
            #         "tags": [],
            #         "url": ds_link}
            data = {"class":"dataset",
            "gitaccess":{"branch":"dell-model-1.5",
            "credentials":{},
            "url":ds_link},
            "name":ds_name,
            "source":"git"}
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
    create_ds(url, user, token, ds_name, ds_link)
    print("[create_resource_job] after ds")
    poll_for_resource_creation(url, user, token, "dataset", ds_name)
    with open('output.txt','w') as out_file:
        out_file.write("create resource job end")
    print("[create_resource_job] finish")
