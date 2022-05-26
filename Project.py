def breast_cancer_prediction_system():
    
    #Importing the modules we need for completion of this project.
    
    import numpy as np #for Converting our array.
    
    import pandas as pd #for reading our dataset
    
    import sklearn.datasets #for getting our dataset
    
    from sklearn.linear_model import LogisticRegression #for prediction
    
    from sklearn.metrics import accuracy_score #for checking our accuracy score
    
    from sklearn.model_selection import train_test_split # for splitting our data into training and testing data.
    
    breast_cancer = sklearn.datasets.load_breast_cancer() #Loading our dataset.
    
    x = breast_cancer.data 
    
    y = breast_cancer.target
    
    print(x.shape, y.shape)
    
    data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    
    data['class'] = breast_cancer.target
    
    print(data.head()) #For printing first Five Rows of our data
        
    print(data['class'].value_counts())
        
    for i in breast_cancer.target_names:
        print(i,end="\n")
    
    print(data.groupby('class').mean())
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.08, stratify=y, random_state=1)
    
    # test size--> for defining how much of data you need
    
    # stratify --> to divide the data equally
    
    # random_state --> specific split of data. each  value of random state divides the data differently
    
    # we are stratifying y because it has only 2 values and therefore can equate the margin
    
    classifier = LogisticRegression() #Algorithm
    
    classifier.fit(x_train, y_train) #fitting our data into the model
    
    predict1 = classifier.predict(x_train) #for predicting results of training data
    
    accuracy = accuracy_score(y_train, predict1) #For checking accuracy on training data
    
    print("Accuracy on training data is ",accuracy) #For predicting results of testing data
    
    accuracy2 = accuracy_score(y_test, classifier.predict(x_test)) #For checking accuracy on testing data
    
    print("Accuracy on testing data is ",accuracy2)
    
    # paste your data in the brackets in input_data
    input_data = (18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368)
    
    input_data = np.asarray(input_data) #Converting tuple to array
    
    input_data1 = input_data.reshape(1, -1)  #Reshaping our 1-D array to 2-D array.
    
    z = classifier.predict(input_data1) #Predicting whether he/she is at benign stage or malignant stage.
    
    if z == [1]:
        return "Benign"
    else:
        return "malignant"

print(breast_cancer_prediction_system())