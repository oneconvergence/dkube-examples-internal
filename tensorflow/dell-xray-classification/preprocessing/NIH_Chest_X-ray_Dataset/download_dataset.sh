#!/bin/bash

# Download NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories
# Total size: 45 GB

TORRENT_FILE=NIH_dataset.torrent

sudo apt-get update
sudo apt-get install -y transmission-cli wget

wget -O $TORRENT_FILE http://academictorrents.com/download/557481faacd824c83fbf57dcf7b6da9383b3235a.torrent

printf "Downloading NIH Chest X-ray Dataset of size 45 GB.\n"
printf "It may take couple of hours depending upon speed of Internet.\n"
# remove torrent cache
rm -rf ~/.config/transmission

transmission-cli -w ./ $TORRENT_FILE

printf "\nDownloading Successfully completed of NIH Chest X-ray Dataset.\n"
rm -rf $TORRENT_FILE
