###############
Troubleshooting
###############

*   If *Dkube deploy* fails to comeup for 15mins. Please check, good network connectivity is required as images are pulled from *dockerhub.io*

**System configuration**

*   Make sure that disk space is sufficient enough on k8s nodes - recommended >100GB for Dkube

    *    This is becuase K8S GCM will start removing unused docker images when disk usage goes beyond **85%** (default value), more details here - https://kubernetes.io/docs/concepts/cluster-administration/kubelet-garbage-collection/#image-collection
*   16GB RAM
*   8 CPUs
*   Drop caches when needed. sync; echo 3 > /proc/sys/vm/drop_caches
