.. _dfabui-installation:


#######################
Dfab UI Installation
#######################

*   Install docker-ce version 18.03.1-ce and docker-compose

*   Download dfab ui docker compose file :download:`docker-compose.yaml<./yamls/docker-compose.yaml>`

*   Update environment variables defined in docker-compose.yaml

    *   REST_SERVER_ENDPOINT, replace <MCPU IP> with MCPU IP
    *   STATS_ENDPOINT, replace <STATS SERVER IP> with UI server IP

*   To bringup dfab ui run

    *   `docker-compose up -d`

.. note::
    docker-compose will ask for docker hub access to download ui docker images


*   To access Dab UI goto browser and type http://<ui server ip>:3000

*   To access Swagger UI goto browser and type http://<mcpu ip>:8080/v1/ui
