.. _kubernetes-installation:


#######################
Kubernetes Installation
#######################


*	**Install Kubernetes version 1.10**

*   nvidia-docker plugin Installation

    *   Install the nvidia docker runtime
    *   Install the nvidia docker plugin
    *   Please follow instructions mentioned @ https://github.com/NVIDIA/k8s-device-plugin
    *   Make sure that **Environment="KUBELET_EXTRA_ARGS=--feature-gates=DevicePlugins=true"** is enabled in **/etc/systemd/system/kubelet.service.d/10-kubeadm.conf**


*   Dkube Device Plugin Installation

    *   Dkube device plugin enables disaggregated GPUs. GPUs are dynamically attached to the pod when it is scheduled and are detached when pod terminates.
    *   GPUs can be requested by POD by using *limits* section in the YAML. Snippet is shown below.
    *   Install Dkube device plugin using below yaml *kubectl apply -f dkube-dev-plugin.yaml*
	*	**Note - PODs container images must be built with CUDA support to be able to use GPU**

.. code-block:: yaml

  containers:
    - name: cuda-container
      image: nvidia/cuda:9.0-devel
      resources:
        limits:
          nvidia.com/gpu: 2 # requesting 2 GPUs
..

    *   *kubectl apply -f dkube-device-plugin.yaml*

:download:`dkube-device-plugin.yaml<./yamls/dkube-device-plugin.yaml>`.

    *   *Or copy and save the text below*

.. literalinclude:: ./yamls/dkube-device-plugin.yaml
   :language: YAML
..
