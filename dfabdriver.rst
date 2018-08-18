.. _dfabdriver-installation:


########################
Dfab Driver Installation
########################

*	**Prerequisites**

	*	MCPU kernel version : 3.10.0-514.el7.x86_64
	*	On MCPU, python version should be 2.7
	*	On MCPU, port 8080 should be available for Dfab
	*	Nvidia driver should be installed on all Hosts

		Reference

		http://www.advancedclustering.com/act_kb/installing-nvidia-drivers-rhel-centos-7/


*	**Install dfab driver package**

	*	Download the package from the following link and install

		*	http://www.oneconvergence.com/products/download-login.php
			
		*	rpm -ivh <DFAB rpm package>


*	**Create host IP and GID mapping**

	Create a file naming “hostidmap” under /etc/k8s_support/ and update all hosts connected to switch with their IP address and GID

	Ex. Suppose host having Ip 192.168.50.225 is connected to switch 0 having host GID 0x10 (16 Decimal) then “hostidmap” should have entry like

	192.168.50.225=16


*	**Create gpuid file**

	*	Power on the host and check the Nvidia driver on Host before assigning the device
	*	Create a file naming “gpuid” under /etc/k8s_support/ and update the BDF and GPU UUIDs of all GPUs connected to the RDKs

	*	For getting the GPU’s UUID, attach the gpu one by one to the host, using dev2host or the UI and run the below command on Host

		*	nvidia-smi --query-gpu=gpu_bus_id,uuid --format=csv

	*	Change the host BDF with MCPU’s BDF and save it in gpuid file.

		Ex. Suppose there is a GPU connected to the MCPU and BDF is 1B:00.0.
		Assign the GPU to the host and run

		nvidia-smi --query-gpu=gpu_bus_id,uuid --format=csv
	
		Output of above command : 
		
		000000:0D:00.0,GPU-1b9945ea-e605-6f11-a3b4-fcf7b0981535
		
		Change the BDF from 0000000:0D:00.0 to 0000000:1B:00.0 and save the entry in gpuid  file.


.. note::

	*	Dev2host command for assigning device

		*	dev2host -a -h hostgid(Hex) -d devicegid(Hex) 

 	*	BDFs will be always in capital letter.

.. successful::
*   On successful installation, you will be able to access rest server UI at **http://<mcpu ip>:8080/v1/ui** and able to try rest apis


*	**Dfab Driver uninstallation**

	rpm -e <DFAB rpm package>
