#bash -x sed.sh
append=dkube-demo
ip=99
ami="ami-0551b6dc0b4079e1d" 
#ami="ami-04b9e92b5572fa0d1"
instance_type="m5a.4xlarge"
region=us-west-2
key=dkube-demo
pem=dkube-demo.pem
max_size=10
desired_capacity=1
version=1.14
USER=ubuntu
DKUBEVERSON="1.4.2"
installpath=$HOME
localuser=dkube
sudo chmod 400 $pem
cp install $installpath
#checking for root or not
if [ $(id -u) = "0" ]; then
          export PATH=$PATH:$HOME/bin
fi
#checking awscli installed or not
command -v aws
if [[ "${?}" -ne 0 ]];then
	echo "awscli not installed"
	echo "Installing aws cli"
	sudo apt-get -y install awscli
	echo "please configure awscli and install aws-iam-authenticator https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html"
	exit 0
fi

#untar
tar -xvf eks-script3.tar

if [[ -e terraform_0.12.9_linux_amd64.zip ]];then
        unzip terraform_0.12.9_linux_amd64.zip
	mv terraform eks-getting-started
fi
#change dir
cd eks-getting-started
sed -i -e "s/demo/$append-&/g" -e "/version *= \"[0-9.]*\"/s/\"[0-9.]*\"/\"$version\"/" eks-cluster.tf 
sed -i "s/demo/$append-&/g" variables.tf 
sed -i -e "s/demo/$append-&/g" -e "/image_id *= \"ami-[a-zA-Z0-9]*\"/s/\"ami-[a-zA-Z0-9]*\"/\"$ami\"/" -e "/instance_type *= \"[a-zA-Z0-9.]*\"/s/\"[a-zA-Z0-9.]*\"/\"$instance_type\"/" -e "/key_name *= \"[a-zA-Z0-9-]*\"/s/\"[a-zA-Z0-9-]*\"/\"$key\"/" -e "/max_size *= [0-9]/s/[0-9]/$max_size/" -e "/desired_capacity *= [0-9]/s/[0-9]/$desired_capacity/" eks-worker-nodes.tf
sed -i "s/demo/$append-&/g" outputs.tf
sed -i "s/1.12/$version/g" variables.tf
sed -i "s/\"us-west-2\"/\"$region\"/" providers.tf
sed -i -e "s/demo/$append-&/g" -e "s/10.0.0.0\/16/$ip.0.0.0\/16/" -e "s/\"10.0.\${count.index}.0\/24\"/\"$ip.0.\${count.index}.0\/24\"/" vpc.tf
rm -rf terraform.tfstate terraform.tfstate.backup 
./terraform init
touch result.txt
./terraform apply -auto-approve -no-color |  tee result.txt

#./terraform apply -auto-approve  -no-color > result.txt
sed -n '/config_map_aws_auth =/,/kubeconfig/p'  result.txt > config_map_aws_auth.yaml
sed -i '1d; $d' config_map_aws_auth.yaml
sed -i '1d' config_map_aws_auth.yaml
sed -n '/kubeconfig =/,//p'  result.txt > kubeconfig
sed -i 's/\x1b\[[0-9;]*m//g' kubeconfig
sed -i '1d' kubeconfig
sed -i '1d' kubeconfig
mkdir -p $HOME/.kube
sudo cp kubeconfig $HOME/.kube/config
sudo chown $localuser:$localuser $HOME/.kube/config
ls -larth $HOME/.kube/config
sudo cp ../$pem $installpath
sudo cp install $installpath
kubectl apply -f config_map_aws_auth.yaml

#docker
command -v docker
if [[ "${?}" -ne 0 ]];then
	VERSIONSTRING="5:18.09.2~3-0~ubuntu-bionic"
	echo "Docker does not exist\n"
	echo "installing Docker\n"
	sudo apt-get remove docker docker-engine docker.io containerd runc
	sudo apt-get -y update
	sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo apt-key fingerprint 0EBFCD88
	sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
	sudo apt-get -y update
	sudo apt-get install docker-ce=$VERSIONSTRING docker-ce-cli=$VERSIONSTRING containerd.io
fi
sudo systemctl start docker
sudo docker login -u lucifer001 -p lucifer@dkube
sudo docker pull ocdr/dkubeadm:1.4.2
sudo docker run --rm -it -v $installpath/.dkube:/root/.dkube ocdr/dkubeadm:1.4.2 init
sudo cp $installpath/install $installpath/.dkube/install
sudo chown -R $localuser:$localuser $HOME/.dkube

#kubectl
command -v kubectl
if [[ "${?}" -ne 0 ]]; then
	echo "Kubectl does not exist\n"
	echo "Installing kubectl\n"
	curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
	chmod +x ./kubectl
	sudo mv ./kubectl /usr/local/bin/kubectl
	kubectl version
fi

sleep 150s
nodes=$(kubectl get no -o wide | awk '{if (NR!=1) {print $1}}')
nodes=($nodes)
echo "$nodes"
externalip=$(kubectl get no -o wide | awk '{if (NR!=1) {print $7}}')
externalip=($externalip)
echo "$externalip"
internalip=$(kubectl get no -o wide | awk '{if (NR!=1) {print $6}}')
internalip=($internalip)
echo "$internalip"
cd $installpath/.dkube
echo "$PWD"
platform=eks
dkubeuser=ocdkube
dkubepass=oc123
user=ubuntu
DISTRO=ubuntu

#sed dkube.ini
sudo sed -i "/^\[REQUIRED\]/,/^PLATFORM=/s/=.*/=$platform/" dkube.ini
sudo sed -i "/^DISTRO=/s/DISTRO=.*/DISTRO=$DISTRO/" dkube.ini
sudo sed -i "/^USERNAME=/s/USERNAME=.*/USERNAME=$dkubeuser/" dkube.ini
sudo sed -i "/^PASSWORD=/s/PASSWORD=.*/PASSWORD=$dkubepass/" dkube.ini
sudo sed -i "s/DKUBE_NODE_NAME=.*/DKUBE_NODE_NAME=$nodes/" dkube.ini
sudo sed -i "s/STORAGE_DISK_NODE=.*/STORAGE_DISK_NODE=$nodes/" dkube.ini
cat dkube.ini

#sed k8s.ini
sudo sed -i "/^\[deployment\]/,/^provider=/s/=.*/=$platform/" k8s.ini
sudo sed -i "/^distro=/s/distro=.*/distro=$DISTRO/" k8s.ini
for((i=0; i<${#externalip[@]};++i));do sed -i "/^\[nodes\]/,/^#/s/#.*/${externalip[i]} ${internalip[i]}/" k8s.ini; done
sudo sed -i "/^\[ssh-user\]/,/^user=/s/=.*/=$user/" k8s.ini
cat k8s.ini
sudo cp /$HOME/.kube/config $installpath/.dkube/kubeconfig

#passwordless
sudo cp $installpath/$pem $installpath/.dkube/
if [[ ! $externalip[0] =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
	for((i=0; i<${#externalip[@]};++i));do sudo cat ssh-rsa.pub|sudo ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR -o UserKnownHostsFile=/dev/null -i ${pem} ubuntu@${externalip[i]} "cat - >> .ssh/authorized_keys"; done
	echo "after copying"
fi

#Dkube Install
cd $installpath/.dkube
ls -larth $installpath/.dkube
sudo chmod 400 $HOME/.dkube/$pem
sudo ./install
