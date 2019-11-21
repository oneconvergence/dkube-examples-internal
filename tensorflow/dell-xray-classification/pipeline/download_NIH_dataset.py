#!/usr/bin/python3

# Download the NIH Chest X-ray Dataset zip files
# import argparse
import os
import sys
import ssl
import tensorflow as tf
from urllib.request import urlretrieve

# Disable SSL verification for urlretrieve
ssl._create_default_https_context = ssl._create_unverified_context

# URLs for the zip files
NIH_DATA_LINKS = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz'
    # 'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    # 'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    # 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    # 'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    # 'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    # 'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    # 'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    # 'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    # 'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    # 'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    # 'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def download():
    import os
    import tensorflow as tf
    from urllib.request import urlretrieve
    import ssl

    # Disable SSL verification for urlretrieve
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # URLs for the zip files
    NIH_DATA_LINKS = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz'
    # 'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    # 'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    # 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    # 'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    # 'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    # 'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    # 'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    # 'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    # 'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    # 'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    # 'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    # create directory
    DATASET_DIR = "{}/".format(os.getenv('DKUBE_INPUT_DATASETS', None))
    print("DATASET_DIR: {}".format(DATASET_DIR))
    # DATASET_DIR = (DATASET_DIR.format(
        # dkube_store_path, username, dataset_name))
    # if not os.path.exists(DATASET_DIR):
        # os.makedirs(DATASET_DIR)
    for idx, link in enumerate(NIH_DATA_LINKS):
        # fn = os.path.join(DATASET_DIR, 'images_%03d.tar.gz' % (idx + 1))
        # print('downloading', fn, '...')
        # urlretrieve(link, fn)  # download the zip file
        fn = DATASET_DIR+'images_%03d.tar.gz'%(idx + 1)
        print("fn : {}".format(fn))
        with tf.gfile.GFile(fn) as file:
            print('downloading', file, '...')
            urlretrieve(link, file)
    print("Download completed successfully and it is available at {}".format(
        DATASET_DIR))

download()
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, help="Dkube UserName")
    parser.add_argument('--auth_token',type=str, help='auth_token')
    parser.add_argument('--access_url', type=str, help='access_url')
    args = parser.parse_args()
    user = args.user
    auth_token = args.auth_token
    access_url = args.access_url
    if not user:
        print("User is not specified, Please provide user with --user argument")
    if not token:
        print("Auth token is not specified, Please provide token with --auth_token argument")
    if not access_url:
        print("Access URL is not specified, Please provide access url with --access_url argument")

    # if not dkube_store_path:
        # dkube_store_path = "/var/dkube/dkube-store"
    download()


if __name__ == '__main__':
    main()
'''
