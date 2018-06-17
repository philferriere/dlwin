from __future__ import print_function
import sys, argparse
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

tracker_length = 30

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # 28x28x32 -> 26x26x32
        x = F.relu(self.conv2(x))      # 26x26x32 -> 24x24x64
        x = F.max_pool2d(x, 2) # 24x24x64 -> 12x12x64
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.view(-1, 12*12*64)       # flatten 12x12x64 = 9216
        x = F.relu(self.fc1(x))        # fc 9216 -> 128
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)                # fc 128 -> 10
        return F.log_softmax(x, dim=1) # to 10 logits

def train(args, model, device, train_loader, optimizer):
    model.train()
    start_time = time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            percentage = 100. * batch_idx / len(train_loader)
            cur_length = int((tracker_length * int(percentage)) / 100)
            bar = '=' * cur_length + '>' + '-' * (tracker_length - cur_length)
            sys.stdout.write('\r{}/{} [{}] - loss: {:.4f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                bar, loss.item()))
            sys.stdout.flush()

    train_time = time() - start_time
    sys.stdout.write('\r{}/{} [{}] - {:.1f}s {:.1f}us/step - loss: {:.4f}'.format(
        len(train_loader.dataset), len(train_loader.dataset), '=' * tracker_length, 
        train_time, (train_time / len(train_loader.dataset)) * 1000000.0, loss.item()))
    sys.stdout.flush()

    return len(train_loader.dataset), train_time, loss.item()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    return test_loss, test_accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=12, metavar='N',
                        help='number of epochs to train (default: 12)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        print("\nEpoch {}/{}".format(epoch, args.epochs))
        train_len, train_time, train_loss = train(args, model, device, train_loader, optimizer)
        test_loss, test_accuracy = test(args, model, device, test_loader)
        sys.stdout.write('\r{}/{} [{}] - {:.1f}s {:.1f}us/step - loss: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}'.format(
            train_len, train_len, '=' * tracker_length, 
            train_time, (train_time / train_len) * 1000000.0, train_loss,
            test_loss, test_accuracy))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
