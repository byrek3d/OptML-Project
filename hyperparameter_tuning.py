from sklearn.model_selection import KFold
def train_with_cross_validation(model, parameters):
    '''
    Perform a K-fold training with CV using the specified parameters.
    This is done by splitting the train set into K-different folds, and performing a training and testing step 
    using the compinations of the folds as new training and validation sets.
    
    This whole procedure is done using the specified parameters for number of folds, loss function, number of epochs, batch size, and optimizer.
    '''
    
    #Define the Kfold to be used when splitting the training dataset
    kfold = KFold(n_splits=parameters["folds"], shuffle = True)
    results = {}
    
    #Itereate K number of times, each time changing the indeces of the train/validation sets
    for fold, (train_index, validation_index) in enumerate(kfold.split(train_dataset)):
        
        print(" -Fold Nr: ", fold+1)
        
        # Create the Training and Validation sets, using the assigned indeces
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_index)
        
        #Use the previous datasets to create DataLoaders with the specified batch_size
        train_loader_Kfold = torch.utils.data.DataLoader(train_dataset, sampler=train_subsampler, batch_size=parameters["batch_size"], pin_memory=use_gpu)
        validation_loader_Kfold  = torch.utils.data.DataLoader(train_dataset, sampler=validation_subsampler, batch_size=parameters["batch_size"], pin_memory=use_gpu)


        torch.cuda.empty_cache()
        train_loader_Kfold , validation_loader_Kfold = send_data_to_gpu(train_loader_Kfold,validation_loader_Kfold)

        #Reset the model weights for the new iteration
        model.apply(reset_weights)
        
        #Define the criterion, optimizer and scheduler as specified in the parameters
        criterion = parameters["criterion"]()

        optimizer = parameters["optimizer"](model.parameters(), lr=parameters["lr"])   
        
        scheduler = parameters["scheduler"](optimizer, **parameters["scheduler_parameters"])
       

        #Repeat the training step 'epoch'-number of times. Keep track of the accuracy when testing on the current validation set
        prev_val_accuracy = 0
        for epoch in range(parameters["epochs"]):
            
            train(model, train_loader_Kfold, criterion, optimizer,parameters["batch_size"])
            
            current_accuracy = 100.0 * (test(model, validation_loader_Kfold,parameters["batch_size"]))

            print(f"      Epoch: {epoch+1}/{parameters['epochs']}, Validation Score: {current_accuracy:.4}")

            '''
            #Early stopping
            if(current_accuracy < prev_val_accuracy):
              break
            '''
            prev_val_accuracy = current_accuracy
            try:
              scheduler.step()
            except:
              scheduler.step(current_accuracy)

        results[fold] = current_accuracy

    # Print fold results
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    avg_acc = sum/len(results.items())
    print(f'   Average validation score of the {parameters["folds"]} folds : {avg_acc:.4} %')
    print('--------------------------------')
        
    return avg_acc

def grid_search(model, parameters):
    '''
    Perform a grid search on the model using the given parameters, calling K-fold cross validation for each combination. 
    After all the combinations of parameters have been tried, return the parameters that had the best average validation accuracy.
    '''

    start_time = time.time()

    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_acc = 0.0
    best_settings = {}
    
    for params in tqdm(permutations_dicts):
        print("-CURRENT PARAMETERS : ", params)
        acc = train_with_cross_validation(model, params)
        
        if acc > best_acc:
            best_acc = acc
            best_settings = params
    print(f"Best overall accuracy {best_acc} for these parameters {best_settings}")
    
    print(f"--- Total time: {((time.time() - start_time)/60):.4} minutes ---" )

    return best_settings, best_acc