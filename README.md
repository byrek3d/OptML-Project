## Optimization for Machine Learning Mini-project (CS-439, EPFL)

**Authors**: Mustapha Ajeghrir, Nicolas Baldwin, Gerald Sula

**Prerequisites**: In order to run the files in this project, some external libraries are required

 - pytorch: [Instructions depend on system](https://pytorch.org)
 - torchvision: conda install -c pytorch torchvision
 - plotly: conda install -c plotly plotly ; conda install -c plotly plotly-orca
 - itertools: conda install -c anaconda more-itertools
 ----- 
 **Project Structure**:
 
 - run.ipynb: This is the main notebook that goes through the training and evaluation phase for all our models and produces the plots included in the report. Please note that running the full notebook does take a considerable amount of time, because of the substantial number of times the models are trained. For this reason it is recommended to use a machine connected with a GPU, to accelerate the computations. 
 - model.py: This file contains the model architecture used throughout the project as well as some function used in combination with this class
 - helpers.py: In this file we have grouped the majority of functions that perform the data loading, training and testing procedure as well as the final statistical analysis
 - plotting.py: This file contains the different functions that are called for creating the plots displayed in the main notebook and report. 
 - hyperparameter_tuning.py: This file contains the functions created for performing hyperparameter tuning through a grid search and k-fold cross validation.
 - 'hyperparameter_tuning' directory:
	 - hyperparameter_tuning.ipynb: In this notebook we have performed the gridsearch of the parameters and used the best ones found here as input for the main notebook.
	 - best_parameters.txt: The list of best parameters for each combination of scheduler and optimizer

 - 'plots' directory: The several plots created in the project will be saved as .png files in this folder
 - 'data' directory: After running the run.ipynb notebook, the data needed for this project will be automatically downloaded in this folder
-----
**Report**: [Link to the PDF](https://github.com/byrek3d/OptML-Project/blob/main/report.pdf)


 
