##################
Dkube Installation
##################

.. highlight:: py


*   **Dkube Github Application**

	*	Register DKube application on github as OAuth app
	*   Reference: https://developer.github.com/apps/building-oauth-apps/creating-an-oauth-app/
	*   Following fields need to be filled while registration,

    	*   Application name : ``DKube``
    	*   Homepage URL : ``http://<DKube UI server IP>:32222/dkube/ui/``
    	*   Authorization callback URL : ``http://<DKube UI server IP>:32222/dkube/ui/callback``
    	*   Replace ``<DKube UI server IP>`` with ``ip address of master-controller`` of the k8s cluster.

.. note::

    ``<DKube UI server IP>`` must be reachable from internet.
    Please see section *Dkube Cloud Installation* when Dkube is installed on public cloud.
..

		*	Upon successful registration of App, Github returns *Client ID* and *Client secret*
		*	These values needs to be passed for *Dkube Installation*

*	**Dkube Installer**

Kubernetes v1.10 must be installed before starting with Dkube. Please refer section :ref:`kubernetes-installation`

.. note::

    Prerequisites,
        - python3.5
        - pip3
        - virtualenv
..

	*	Installation of *Dkube* and the all the required softwares is automated by the *Dkube Installer*
	*   Checkout *Dkube Installer* as ``git clone https://github.com/mak-454/dkube-install``
	*	Create a virtual environment ``virtualenv -p python3 dkube-installation``
	*	Activate the virtual environment ``source dkube-installation/bin/activate``
	*   ``cd installer``
	*	Install all the *Dkube Installer* requirements ``pip3 install -r requirements.txt``

.. warning::

	This is an important point before proceeding further.
	When *dkube* is deployed, it pulls the required installation files from github, by default *github rate limit* prevents
	making number of requests per hour. The rate limit is much higher when used with github personal access token. Hence for the *dkube deploy* to
	be successful, it is recommended that the person installing should generate a GIT private token and provide it during installation.
	The private token is shortlived and neither *dkube* installer nor *dkube* software uses it. User can always delete the token later for security reasons.

	Use the link https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/ to create a personal access token.

	Execute ``export GITHUB_TOKEN=<github-personal-access-token>`` in the shell from which the *dkube* will be *deployed/deleted*
..

*   **Deploy Dkube In Cluster**

    *   Run the dkube installer utility *dkubectl* as ``./dkubectl deploy --pkg all --client_id <Client-ID> --client_secret <Client-Secret>``
    *   <*Client-ID*> is the Client ID returned by github when *Dkube* App is registered.
    *   <*Client-Secret*> is the Client Secret returned by github when *Dkube* App is registered.

*   **Deploy Dkube In Cloud**
    
    *   For Amazon AWS,

        *   Expose the *master* node/VM of the k8s cluster to external traffic by connecting it to *external network*

    *   For gCloud,

        *   Create an *k8s ingress controller* with *cloud balancer*

    *   ``Port 32222`` must be allowed in *Security groups* of cloud.
    *   Configure the returned *Public IP* as *<DKube UI server IP>* parameter while registering Dkube App.
    *   Deploy the *dkube* as ``./dkubectl deploy --pkg all --client_id <Client-ID> --client_secret <Client-Secret> --external_access_ip <IP>``
    
    |


    *   The command also monitors the creation of all *Dkube* resources for 15minutes. It displays the monitor progress bar.
    *   At the end of installation, the component wise installation status and an overall installation status is displayed.
    *   If any of the component does not move to *Active* state in 15minutes then the installation is declared *Failure*.
    *   User may want to  *Delete Dkube* and reinstall again.

.. seealso::

    Section: *Delete Dkube*
..

    *   Following image shows the reference for successful *deploy*.

.. image:: images/dkube_deploy_success.png
   :width: 400px
   :height: 100px
   :scale: 400 %
   :alt: alternate text
   :align: center
..

*   **Onboard User**

	*	After the successful deployment of *Dkube*, users need to be onboarded onto *Dkube* platform.
	*	User onboard process is completely automated with *dkubectl* utility.
	*	Execute the command ``./dkubectl onboard --git-username <lucifer>``
	*	The username is validated with *github* when user login into *Dkube* so the onboarded name has to match with *github name* of the user.
	
*   **Delete Dkube**

    *   Run the dkube installer utility *dkubectl* as ``./dkubectl delete --all``
    *   The command will delete the *dkube* and all the dependencies installed.
    *   It also monitors for the successful deletion of all the resources for the maximum of 15minutes.

.. warning::

    All the persistent data will also be deleted. When reinstalled, *dkube* will be deployed afresh without any history information.
..

    *   Following image shows the reference for successful *delete*.

.. image:: images/dkube_delete_success.png
   :width: 400px
   :height: 100px
   :scale: 400 %
   :alt: alternate text
   :align: center
..


.. tip:: 

    Happy working with *Dkube*!
