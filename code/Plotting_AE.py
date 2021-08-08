def sep_list(inceptionList):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for i in inceptionList:
        list1.append(i[0])
        list2.append(i[1])
        list3.append(i[2])
        list4.append(i[3])
    return list1, list2, list3, list4

def module_check(module):
    import sys
    if module not in sys.modules:
        return True
    else:
        return False


def residual_plot (model,test_data,train_data):
    if module_check('matplotlib.pyplot'):
        import matplotlib.pyplot as plt
        print("importing matplot")
    
    import torch
    if module_check('numpy'):
        import numpy as np
    #Turn data into a tensor
    test_data = torch.tensor(test_data.values, dtype=torch.float)
    train_data = torch.tensor(train_data.values, dtype=torch.float)

    #Have the model predict and convert to numpy array
    prediction_test = model(test_data)
    prediction_train = model(train_data)
    prediction_test = prediction_test.detach().numpy()
    test_data = test_data.detach().numpy()
    prediction_train = prediction_train.detach().numpy()
    train_data = train_data.detach().numpy()

    #Calculate the residuals
    residual_test = test_data - prediction_test
    residual_train = train_data - prediction_train

    #Seperating data 
    residual_test_pt, residual_test_eta, residual_test_phi, residual_test_mass = sep_list(residual_test)
    residual_train_pt, residual_train_eta, residual_train_phi, residual_train_mass = sep_list(residual_train)
    train_pt, train_eta, train_phi, train_mass = sep_list(train_data)
    test_pt, test_eta, test_phi, test_mass = sep_list(test_data)

    #Plotting the scatter plots
    #This plots the absolute value, to catch bad data
    print("These are the scatter plots")
    plt.scatter(train_pt, np.abs(residual_train_pt))
    plt.title("Train Data pT")
    plt.show()
    plt.scatter(train_eta, np.abs(residual_train_eta))
    plt.title("Train Data eta")
    plt.show()
    plt.scatter(train_phi, np.abs(residual_train_phi))
    plt.title("Train Data phi")
    plt.show()
    plt.scatter(train_mass, np.abs(residual_train_mass))
    plt.title("Train Data mass")
    plt.show()
    plt.scatter(test_pt, np.abs(residual_test_pt))
    plt.title("Test Data pT")
    plt.show()
    plt.scatter(test_eta, np.abs(residual_test_eta))
    plt.title("Test Data eta")
    plt.show()
    plt.scatter(test_phi, np.abs(residual_test_phi))
    plt.title("Test Data phi")
    plt.show()
    plt.scatter(test_mass, np.abs(residual_test_mass))
    plt.title("Test Data mass")
    plt.show()

    #Plotting Histograms
    print("These are the histograms")
    plt.hist(residual_train_pt, 50)
    plt.title("Train pT")
    plt.show()
    plt.hist(residual_train_eta, 50)
    plt.title("Train eta")
    plt.show()
    plt.hist(residual_train_phi,50)
    plt.title("Train phi")
    plt.show()
    plt.hist(residual_train_mass,50)
    plt.title("Train mass")
    plt.show()
    plt.hist(residual_test_pt,50)
    plt.title("Test pT")
    plt.show()
    plt.hist(residual_test_eta,50)
    plt.title("Test eta")
    plt.show()
    plt.hist(residual_test_phi,50)
    plt.title("Test phi")
    plt.show()
    plt.hist(residual_test_mass,50)
    plt.title("Test mass")
    plt.show()

    
