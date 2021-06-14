import matplotlib.pyplot as plt
from pandas import DataFrame
import plotly.express as px
import plotly.graph_objects as go
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, LambdaLR
import torch
import torchvision
from model import *
from helpers import train, test

#function to plot losses and accuracies
def show_stats(models_results, experiment_name):
  train_losses = []
  test_losses = []
  train_accs = []
  test_accs = []

  ### PLOTING LOSSES

  for key in models_results.keys():
    print(key)
    train_curve = list(zip(models_results[key]['avg_train_losses'],range(1,len(models_results[key]['avg_train_losses'])+1)))
    validation_curve = list(zip(models_results[key]['avg_test_losses'],range(1,len(models_results[key]['avg_test_losses'])+1)))
    train = list(map(lambda x: [key,x[0],x[1]],train_curve))
    validation = list(map(lambda x: [key,x[0],x[1]],validation_curve))
    train_losses.extend(train)
    test_losses.extend(validation)
    
  df = DataFrame(train_losses,columns = ['Model','Loss','Epoch'])
  fig = px.line(df, x="Epoch", y="Loss", title=f'Train Losses Plot {experiment_name}', color='Model', width = 1400, height=800)
  fig.update_layout(legend=dict(
      font=dict(
              size=20,
              color="black"
          ),
      yanchor="top",
      y=0.8,
      xanchor="left",
      x=0.63
  ))
  fig.show()
  fig.write_image(f"plots/train_loss_{experiment_name.replace(' ', '_')}.png")

  df = DataFrame(test_losses,columns = ['Model','Loss','Epoch'])
  fig = px.line(df, x="Epoch", y="Loss", title=f'Test Losses Plot {experiment_name}', color='Model', width = 1400, height=800)
  fig.update_layout(legend=dict(
      font=dict(
              size=20,
              color="black"
          ),    
      yanchor="top",
      y=0.8,
      xanchor="left",
      x=0.63
  ))
  fig.show()
  fig.write_image(f"plots/test_loss_{experiment_name.replace(' ', '_')}.png")

  ### PLOTING ACCURACIES

  
  for key in models_results.keys():
    train_curve = list(zip(models_results[key]['avg_train_accuracies'],range(1,len(models_results[key]['avg_train_accuracies'])+1)))
    validation_curve = list(zip(models_results[key]['avg_test_accuracies'],range(1,len(models_results[key]['avg_test_accuracies'])+1)))
    train = list(map(lambda x: [key,x[0],x[1]],train_curve))
    validation = list(map(lambda x: [key,x[0],x[1]],validation_curve))
    train_accs.extend(train)
    test_accs.extend(validation)


  df2 = DataFrame(train_accs,columns = ['Model','Accuracy','Epoch'])
  fig2 = px.line(df2, x="Epoch", y="Accuracy", title=f'Train Accuracy Plot {experiment_name}', color='Model', width = 1400, height=800)
  # fig2 = px.line(df2, x="Epoch", y="Accuracy", title='Train Accuracy Plot', color='Model', width = 1400, height=800)
  fig2.update_layout(legend=dict(
      font=dict(
              size=20,
              color="black"
          ),    
      yanchor="top",
      y=0.30,
      xanchor="left",
      x=0.63
  ))
  fig2.show()
  fig2.write_image(f"plots/train_acc_{experiment_name.replace(' ', '_')}.png")

  df2 = DataFrame(test_accs,columns = ['Model','Accuracy','Epoch'])
  # fig2 = px.line(df2, x="Epoch", y="Accuracy", title='Test Accuracy Plot', color='Model', width = 1400, height=800)
  fig2 = px.line(df2, x="Epoch", y="Accuracy", title=f'Test Accuracy Plot {experiment_name}', color='Model', width = 1400, height=800)
  fig2.update_layout(legend=dict(
      font=dict(
              size=20,
              color="black"
          ),
      yanchor="top",
      y=0.30,
      xanchor="left",
      x=0.63
  ))
  fig.show()
  fig2.write_image(f"plots/test_acc_{experiment_name.replace(' ', '_')}.png")

  for key in models_results.keys():
    print('*************************************')
    print(f'{key}')
    print('average train accuracy:')
    print(models_results[key]['avg_train_accuracy'])
    print('std train accuracy:') 
    print(models_results[key]['final_train_acc_std'])
    print()
    print('average test accuracy:')
    print(models_results[key]['avg_test_accuracy'])
    print('std test accuracy:')
    print(models_results[key]['final_test_acc_std'])
    print()
    print('average train loss:') 
    print(models_results[key]['avg_final_train_loss'])
    print('std train loss:') 
    print(models_results[key]['final_train_loss_std'])
    print()
    print('average test loss:')
    print(models_results[key]['avg_final_test_loss'])
    print('std test loss:')
    print(models_results[key]['final_test_loss_std'])


# Used only for plotting how the schedulers look like
EPOCHS = 25
import warnings

def plot_stepLR():
  '''
  Plot the progression of the learning rate when using StepLR
  '''
  
  warnings.filterwarnings("ignore") 

  plotting_model = Net()

  scheduler1 = StepLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 2, 0.5)
  scheduler2 = StepLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 4, 0.5)
  scheduler3 = StepLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 4, 0.25)
  scheduler4 = StepLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 6, 0.25)

  lr_evol1 = [0]*EPOCHS
  lr_evol2 = [0]*EPOCHS
  lr_evol3 = [0]*EPOCHS
  lr_evol4 = [0]*EPOCHS

  for epoch in range(EPOCHS):
    lr_evol1[epoch] = scheduler1.get_last_lr()[0]
    lr_evol2[epoch] = scheduler2.get_last_lr()[0]
    lr_evol3[epoch] = scheduler3.get_last_lr()[0]
    lr_evol4[epoch] = scheduler4.get_last_lr()[0]

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()

  plt.plot(lr_evol1, label="$\gamma = 0.5$, $step = 2$")
  plt.plot(lr_evol2, label="$\gamma = 0.5$, $step = 4$")
  plt.plot(lr_evol3, label="$\gamma = 0.25$, $step = 4$")

  plt.legend()
  plt.title("StepLR")
  plt.xlabel("Epochs")
  plt.ylabel("Lr")
  plt.savefig("./plots/step_scheduler.png", dpi=100)
  plt.show()


def plot_cosineLR():
  '''
  Plot the progression of the learning rate when using CosineAnnealingLR
  '''

  warnings.filterwarnings("ignore") 

  plotting_model = Net()

  scheduler1 = CosineAnnealingLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 3, eta_min=0.01)
  scheduler2 = CosineAnnealingLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 8, eta_min=0.01)
  scheduler3 = CosineAnnealingLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 12, eta_min=0.01)
  scheduler4 = CosineAnnealingLR(torch.optim.SGD(plotting_model.parameters(), lr=1), 15, eta_min=0.01)

  lr_evol1 = [0]*EPOCHS
  lr_evol2 = [0]*EPOCHS
  lr_evol3 = [0]*EPOCHS
  lr_evol4 = [0]*EPOCHS

  for epoch in range(EPOCHS):
    
    lr_evol1[epoch] = scheduler1.get_last_lr()[0]
    lr_evol2[epoch] = scheduler2.get_last_lr()[0]
    lr_evol3[epoch] = scheduler3.get_last_lr()[0]
    lr_evol4[epoch] = scheduler4.get_last_lr()[0]

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()

  plt.plot(lr_evol1, label="$T_{max} = 3$")
  plt.plot(lr_evol2, label="$T_{max} = 8$")
  plt.plot(lr_evol3, label="$T_{max} = 12$")

  plt.legend()
  plt.title("Cosine Annealing")
  plt.xlabel("Epochs")
  plt.ylabel("Lr")
  plt.savefig("./plots/cosine_scheduler.png", dpi=100)
  plt.show()

def plot_lambdaLR():
  '''
  Plot the progression of the learning rate when using LambdaLR
  '''

  warnings.filterwarnings("ignore") 

  plotting_model = Net()

  def lr_lambda_test_1(epoch):
    return 1/((epoch)//3+1)  #For each epoch, this will be multiplied by the learning rate
  def lr_lambda_test_2(epoch):
    return 1/((epoch)//3+1)**2  #For each epoch, this will be multiplied by the learning rate
  def lr_lambda_test_3(epoch):
    return 1/((epoch)//4+1)**3  #For each epoch, this will be multiplied by the learning rate

  scheduler1 = LambdaLR(torch.optim.SGD(plotting_model.parameters(), lr=1), lr_lambda_test_1 )
  scheduler2 = LambdaLR(torch.optim.SGD(plotting_model.parameters(), lr=1), lr_lambda_test_2 )
  scheduler3 = LambdaLR(torch.optim.SGD(plotting_model.parameters(), lr=1), lr_lambda_test_3 )

  lr_evol1 = [0]*EPOCHS
  lr_evol2 = [0]*EPOCHS
  lr_evol3 = [0]*EPOCHS

  for epoch in range(EPOCHS):
    
    lr_evol1[epoch] = scheduler1.get_last_lr()[0]
    lr_evol2[epoch] = scheduler2.get_last_lr()[0]
    lr_evol3[epoch] = scheduler3.get_last_lr()[0]

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()

  plt.plot(lr_evol1, label="$\lambda(x) = \dfrac{1}{x//3 +1}$")
  plt.plot(lr_evol2, label="$\lambda(x) = (\dfrac{1}{x//3 +1})^2$")
  plt.plot(lr_evol3, label="$\lambda(x) = (\dfrac{1}{x//4 +1})^3$")

  plt.legend()
  plt.title("LambdaLR")
  plt.xlabel("Epochs")
  plt.ylabel("Lr")

  plt.savefig("./plots/lambda_scheduler.png", dpi=100)
  plt.show()


def plot_reduceOnPLateuLR(device):
  '''
  Plot the progression of the learning rate when using reduceOnPLateuLR. 
  This one is different as it requires actually training a model to see the progression of training. 
  '''

  train_dataset =  torchvision.datasets.MNIST('./data/files/', train=True, download=True,  transform=torchvision.transforms.Compose([
                                                                                       torchvision.transforms.ToTensor()]))
  test_dataset =   torchvision.datasets.MNIST('./data/files/', train=False, download=True,  transform=torchvision.transforms.Compose([
                                                                                       torchvision.transforms.ToTensor()]))

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

  train_loader , test_loader = send_data_to_gpu(train_loader, test_loader, device)

  warnings.filterwarnings("ignore") 

  acc_sgd_red = [0]*EPOCHS

  lr_evol1 = [0]*EPOCHS

  FOLDS = 1
  for fold in range(FOLDS):

    plotting_model = Net()
    plotting_model.to(device)

    plotting_optimizer = torch.optim.SGD(plotting_model.parameters(), lr=0.4)

    scheduler = ReduceLROnPlateau(plotting_optimizer,patience=2, mode='max', factor= 0.3)

    for epoch in range(EPOCHS):

        lr_evol1[epoch] = scheduler.optimizer.param_groups[0]["lr"]

        train(plotting_model, train_loader, torch.nn.CrossEntropyLoss(), plotting_optimizer,64)
        acc = test(plotting_model, test_loader,64)
        acc_sgd_red[epoch] += acc
        
        scheduler.step(acc)
        


    

  for i in range(EPOCHS): #Dividing by number of folds to get avg
    acc_sgd_red[i] /= FOLDS
      

  acc_sgd_red = [0]*EPOCHS

  lr_evol2 = [0]*EPOCHS

  FOLDS = 1
  for fold in range(FOLDS):


    plotting_model = Net()
    plotting_model.to(device)

    plotting_optimizer = torch.optim.SGD(plotting_model.parameters(), lr=0.4)

    scheduler = ReduceLROnPlateau(plotting_optimizer,patience=1, mode='max', factor= 0.8)

    for epoch in range(EPOCHS):

        lr_evol2[epoch] = scheduler.optimizer.param_groups[0]["lr"]

        train(plotting_model, train_loader, torch.nn.CrossEntropyLoss(), plotting_optimizer,64)
        acc = test(plotting_model, test_loader,64)
        acc_sgd_red[epoch] += acc
        
        scheduler.step(acc)
        


    

  for i in range(EPOCHS): #Dividing by number of folds to get avg
    acc_sgd_red[i] /= FOLDS
      



  plt.plot(lr_evol1, label= "$factor = 0.3$ $patience =2$")
  plt.plot(lr_evol2, label= "$factor = 0.8$ $patience =1$")
  plt.legend()
  plt.title("ReduceOnPlateau")
  plt.xlabel("Epochs")
  plt.ylabel("Lr")

  plt.savefig("./plots/reduceOnPlateu_scheduler.png", dpi=100)
  plt.show()