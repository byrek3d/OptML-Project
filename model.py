import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(28*28, 256) 
        self.linear2 = nn.Linear(256, 100) 
        self.final = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

def reset_weights(m):
    '''
    Reset model weights to avoid weight leakage. Useful during hyperparameter tuning
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def send_data_to_gpu(train_loader, test_loader, device):
    '''
    load data to gpu (loads all data to gpu, if available, in order speedup crossvalidation)
    we were able to load all the data all at once because the MNIST dataset is small enough
    '''

    train_tensor = [[],[]]
    for elem in train_loader:
        train_tensor[0].append(elem[0])
        train_tensor[1].append(elem[1])
    train_tensor[0] = torch.cat(train_tensor[0],0).to(device)
    train_tensor[1] = torch.cat(train_tensor[1],0).to(device)

    test_tensor = [[],[]]
    for elem in test_loader:
        test_tensor[0].append(elem[0])
        test_tensor[1].append(elem[1])
    test_tensor[0] = torch.cat(test_tensor[0],0).to(device)
    test_tensor[1] = torch.cat(test_tensor[1],0).to(device)

    return train_tensor,test_tensor
