#!/usr/bin/python3

# Download the NIH Chest X-ray Dataset zip files
import argparse
import os
import sys

from urllib.request import urlretrieve

# URLs for the zip files
NIH_DATA_LINKS = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def download(username, dataset_name, dkube_store_path):
    # create directory
    DATASET_DIR = "{}/dkube/users/{}/dataset/{}"
    dataset_dir = (DATASET_DIR.format(
        dkube_store_path, username, dataset_name))
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for idx, link in enumerate(NIH_DATA_LINKS):
        fn = os.path.join(dataset_dir, 'images_%03d.tar.gz' % (idx + 1))
        print('downloading', fn, '...')
        urlretrieve(link, fn)  # download the zip file
    print("Download completed successfully and it is available at {}".format(
        dataset_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', help="Dkube UserName")
    parser.add_argument('--dataset_name',
                        help='Name of the dataset to be created.')
    parser.add_argument('--dkube_store_path', help='path to dkube store')
    args = parser.parse_args()
    username = args.user
    dataset_name = args.dataset_name
    dkube_store_path = args.dkube_store_path

    if not username:
        print("usage: %s [-h] [--user USER] [--dataset-name DATASET_NAME]" % (
            sys.argv[0]))
        return

    if not dataset_name:
        print("usage: %s [-h] [--user USER] [--dataset-name DATASET_NAME]" % (
            sys.argv[0]))
        return
    if not dkube_store_path:
        dkube_store_path = "/var/dkube/dkube-store"
    download(username, dataset_name, dkube_store_path)


if __name__ == '__main__':
    main()
