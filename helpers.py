import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from model import Net, reset_weights, send_data_to_gpu


scaled_mean= 0.13062754273414612
scaled_std= 0.30810779333114624



def train(model, dataloader, criterion, optimizer,mini_batch_size):
    '''
    Train the given model, using the specified train data (dataloader), loss (criterion) and optimizer
    '''
    
    for b in range(0, dataloader[0].size(0), mini_batch_size):
        if b+mini_batch_size < dataloader[0].shape[0]:
           batch_size = mini_batch_size
        else:
          batch_size = dataloader[0].shape[0] - b
        
        y_pred = model(dataloader[0].narrow(0, b, batch_size))
        
        loss = criterion(y_pred, dataloader[1].narrow(0, b, batch_size))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




def test(model, dataloader,mini_batch_size):
    '''
    Evaluate the accuracy of the given model, tested on the specified test set (dataloader)
    '''
    correct = 0.0
    total = 0.0
    for b in range(0, dataloader[0].size(0), mini_batch_size):

        ##if remaining batch is smaller than mini_batch_size, take last batch in a smaller batch size
        if b+mini_batch_size < dataloader[0].shape[0]:
           batch_size = mini_batch_size
        else:
          batch_size = dataloader[0].shape[0] - b
        
        with torch.no_grad():
            preds = model(dataloader[0].narrow(0, b, batch_size))           # output of linear
            probs = F.softmax(preds, dim=1) # probability distribution
            preds = probs.argmax(dim=1)      # most probable class (for each sample in the batch)

            correct += (preds ==  dataloader[1].narrow(0, b, batch_size)).sum()
            total += len(preds)
            
    return correct / total # accuracy




def compute_stats(train_losses,test_losses,train_accuracies,test_accuracies,train_accuracy,test_accuracy,final_train_loss,final_test_loss,n_runs):
  '''
  Function that computes the means and std of accuracies and losses
  '''
  avg_train_losses = list(map(lambda x: x/n_runs,train_losses))
  avg_test_losses = list(map(lambda x: x/n_runs,test_losses))
  avg_train_accuracies = list(map(lambda x: x/n_runs,train_accuracies))
  avg_test_accuracies = list(map(lambda x: x/n_runs,test_accuracies))

  avg_train_accuracy = sum(train_accuracy)/n_runs
  avg_test_accuracy = sum(test_accuracy)/n_runs
  avg_final_train_loss = sum(final_train_loss)/n_runs
  avg_final_test_loss = sum(final_test_loss)/n_runs

  final_train_loss_std =  sum(list(map(lambda x: (x-avg_final_train_loss)**2,final_train_loss)))/n_runs
  final_test_loss_std =  sum(list(map(lambda x: (x-avg_final_test_loss)**2,final_test_loss)))/n_runs
  final_train_acc_std = sum(list(map(lambda x: (x-avg_train_accuracy)**2,train_accuracy)))/n_runs
  final_test_acc_std = sum(list(map(lambda x: (x-avg_test_accuracy)**2,test_accuracy)))/n_runs
  return (avg_train_losses,avg_test_losses,avg_train_accuracies,avg_test_accuracies,avg_train_accuracy,avg_test_accuracy,avg_final_train_loss,avg_final_test_loss,final_train_loss_std,final_test_loss_std,final_train_acc_std,final_test_acc_std)


#function that trains a model n times and returns information on the loss and accuracies
def train_with_stats(model, parameters, use_gpu, device, n_runs = 10):
  test_losses = []
  train_losses = []
  test_accs = []
  train_accs = []
  final_train_accuracies = []
  final_test_accuracies = []
  final_test_losses = []
  final_train_losses = []

  for i in range(n_runs):

    print(f'train run {i+1}')
    
    #Reset the model weights for the new iteration
    model.apply(reset_weights)

    torch.cuda.empty_cache()

    criterion = parameters["criterion"]()

    optimizer = parameters["optimizer"](model.parameters(), lr=parameters["lr"])   
          
    scheduler = parameters["scheduler"](optimizer, **parameters["scheduler_parameters"])

    prev_val_accuracy = 0

    train_dataset =  torchvision.datasets.MNIST('./data/files/', train=True, download=True,  transform=torchvision.transforms.Compose([
                                                                                        torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((scaled_mean,), (scaled_std,))]))
    test_dataset =   torchvision.datasets.MNIST('./data/files/', train=False, download=True,  transform=torchvision.transforms.Compose([
                                                                                        torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((scaled_mean,), (scaled_std,))])) 
    
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True, pin_memory=use_gpu)

    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=parameters["batch_size"], shuffle=True, pin_memory=use_gpu)

    train_data , test_data = send_data_to_gpu(train_data, test_data, device)

    model = model.to(device)
    
    prev_val_accuracy = 0.0
    for epoch in range(parameters["epochs"]):
        
        train(model, train_data, criterion, optimizer,parameters["batch_size"])
        
        current_accuracy = 100.0 * (test(model, test_data,parameters["batch_size"]))
        
        prev_val_accuracy = current_accuracy
        try:
          scheduler.step()
        except:
          scheduler.step(current_accuracy)
      
        with torch.no_grad():

          train_accuracy = test(model, train_data,parameters["batch_size"]).item()

          test_accuracy = test(model, test_data,parameters["batch_size"]).item()

          y_pred_train = model(train_data[0])

          y_pred_test = model(test_data[0])

          train_loss = parameters["criterion"]()(y_pred_train, train_data[1]).item()

          test_loss = parameters["criterion"]()(y_pred_test, test_data[1]).item()

          model.zero_grad()

          #log current losses and accuracies
          if i == 0:
            test_losses.append(test_loss)
            train_losses.append(train_loss)
            test_accs.append(test_accuracy)
            train_accs.append(train_accuracy)
          else: 
            test_losses[epoch] += test_loss
            train_losses[epoch] += train_loss
            test_accs[epoch] += test_accuracy
            train_accs[epoch] += train_accuracy
          
          if epoch == parameters["epochs"]-1:
            final_train_accuracies.append(train_accuracy)
            final_test_accuracies.append(test_accuracy)
            final_test_losses.append(test_loss)
            final_train_losses.append(train_loss)
    print(f'final_train_acc: {final_train_accuracies[-1]}')
    print(f'final_test_acc: {final_test_accuracies[-1]}')
  
  avg_train_losses,avg_test_losses,avg_train_accuracies,avg_test_accuracies, \
  avg_train_accuracy,avg_test_accuracy,avg_final_train_loss,avg_final_test_loss, \
  final_train_loss_std,final_test_loss_std,final_train_acc_std,final_test_acc_std = \
  compute_stats(train_losses,test_losses,train_accs, test_accs,final_train_accuracies,final_test_accuracies, final_train_losses,final_test_losses,n_runs)
  return avg_train_losses,avg_test_losses,avg_train_accuracies,avg_test_accuracies,avg_train_accuracy,avg_test_accuracy,avg_final_train_loss,avg_final_test_loss,final_train_loss_std,final_test_loss_std,final_train_acc_std,final_test_acc_std


# Helper functions of the LambdaLR scheduler
def lr_lambda_1(epoch):
  return 1  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_2(epoch):
  return 1/((epoch)//3+1)  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_3(epoch):
  return 1/((epoch)//3+1)**2  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_4(epoch):
  return 1/((epoch)//4+1)**2  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_5(epoch):
  return 1/((epoch)//4+1)**3  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_6(epoch):
  return 1/((epoch)//5+1)**3  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_7(epoch):
  return np.exp(-epoch/4)  #For each epoch, this will be multiplied by the learning rate
def lr_lambda_8(epoch):
  return np.exp(-epoch/7)  #For each epoch, this will be multiplied by the learning rate

