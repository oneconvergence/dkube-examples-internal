from __future__ import print_function
import argparse
import os, shutil
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import requests

CLASS_FILE = 'class_file/Net.py'
MODEL_DIR = "/opt/dkube/output"
DATA_DIR = "/opt/dkube/input"
BATCH_SIZE = int(os.getenv('BATCHSIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 5))
print ("ENV, EXPORT_DIR:{}, DATA_DIR:{}".format(MODEL_DIR, DATA_DIR))
# Tensorboard config
# Writer will output to ./runs/ directory by default
logs_dir = "{}/eval".format(MODEL_DIR)
os.system("mkdir -p %s" % logs_dir)
writer = SummaryWriter(log_dir=logs_dir)
global_steps = 1

def log_metrics(key, value, epoch, step):
    url = "http://dkube-exporter.dkube:9401/mlflow-exporter"
    train_metrics = {}
    train_metrics['mode']="train"
    train_metrics['key'] = key
    train_metrics['value'] = value
    train_metrics['epoch'] = epoch
    train_metrics['step'] = step
    train_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
    train_metrics['run_id']=os.getenv('DKUBE_JOB_UUID')
    train_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
    requests.post(url, json = train_metrics)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    global global_steps
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar("Loss/train", loss.item(), global_step=global_steps)
            global_steps += 1



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, accuracy={}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("Accuracy/test", 100. * correct / len(test_loader.dataset))
    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args("")
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if DATA_DIR is None:
        print("No input dataset specified. Exiting...")
        return 1

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_loss = 0
    test_acc = 0
    step = 1
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_metrics = test(args, model, device, test_loader)
        log_metrics('loss', test_metrics[0], epoch, step)
        log_metrics('accuracy', test_metrics[1], epoch, step)
        step += 1
        scheduler.step()

    if args.save_model:
        model_path = '{}/model.pt'.format(MODEL_DIR)
        torch.save(model.state_dict(), model_path)
        shutil.copyfile(CLASS_FILE , os.path.join(MODEL_DIR,CLASS_FILE.split('/')[-1]))

if __name__ == '__main__':
    print("Training mnist model using pytorch:")
    main()

