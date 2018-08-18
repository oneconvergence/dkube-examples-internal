#############################
Dfab Components Installation
#############################

Follow the below given steps on MCPU to start rest serever and UI server

1. Download rpm from the following location and install

https://oneconvergence.box.com/s/hgnzs5xhzuvgz4ipfddh7a5mqt960wbb

rpm -ivh DFAB-1-1.el7.noarch.rpm

2. Install docker

https://docs.docker.com/install/linux/docker-ce/centos/#install-docker-ce-1 (18.03.1-ce)

3. Install docker-compose

sudo pip install docker-compose


4. Create docker-compose.yaml and .env in same directory with the following content.

**docker-compose.yaml**

.. code-block:: yaml

	version: "2"

	services:
	  dfabricinfluxd:
		image: ${DOCKER_DFABRIC_INFLUXDB_IMAGE}
		container_name: dfabricinfluxd
		environment:
		  REST_SERVER_ENDPOINT: ${REST_SERVER_ENDPOINT}
		volumes:
		  - "${INFLUXDB_VOLUME_HOST}:/var/lib/influxdb"
		ports:
		  - "8086:8086"

	  dfabricuiserver:
		image: ${DOCKER_DFABRIC_UISERVER_IMAGE}
		container_name: dfabricuiserver
		environment:
		  REST_SERVER_ENDPOINT: ${REST_SERVER_ENDPOINT}
		ports:
		  - "3000:3000"

#.. include:: ./dfab.yaml


**ENVIRONMENT VARIABLES**

Copy paste the below text in '.env' and update REST_SERVER_ENDPOINT with rest server endpoint

.. code-block:: none

    # influxdb database data volume
    INFLUXDB_VOLUME_HOST=/var/lib/influxdb/

    # dfabric UI server container image
    DOCKER_DFABRIC_UISERVER_IMAGE=oneconvergence123/dfabric-uiserver:v1

    # influxdb container image
    DOCKER_DFABRIC_INFLUXDB_IMAGE=oneconvergence123/dfabric-influxdb:v1

    # REST SERVER ENDPOINT
    REST_SERVER_ENDPOINT=<Update with DFabric_Rest_Server_ip_address:port>

5. Login to docker hub to have access to repo

6. Run the following command to start ui server
    docker-compose up -d

7. To access ui goto browser and type <UI server IP>:3000
