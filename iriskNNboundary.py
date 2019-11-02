import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from scipy.spatial.distance import cdist
from scipy import stats
plt.style.use("seaborn")  # Use GGPlot style for graph


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
np.random.seed(1)

# Setup data
D = np.genfromtxt('code/iris.csv', delimiter=',')
X_train = D[:, 0:2]   # feature
y_train = D[:, -1]    # label

# Setup meshgrid
x1, x2 = np.meshgrid(np.arange(2,5,0.01), np.arange(0,3,0.01))
X12 = np.c_[x1.ravel(), x2.ravel()]

# Compute 1NN decision
#k = 1
#decision = knn_predict(X12, X_train, y_train, k)

#  KNN function
def knn_predict(X_test, X_train, y_train,k):
    n_X_test = X_test.shape[0]
    decision = np.zeros((n_X_test, 1))
    for i in range(n_X_test):
        point = X_test[[i],:]

        #  compute euclidan distance from the point to all training data
        dist = cdist(X_train, point)

        #  sort the distance, get the index
        idx_sorted = np.argsort(dist, axis=0)

        #  find the most frequent class among the k nearest neighbour
        pred = stats.mode( y_train[idx_sorted[0:k]] )

        decision[i] = pred[0]
    return decision

#Function to flip the values and select random values for each train label and return the training set
def flip_function(array,flip):
    for i in range(flip):
        index = np.random.choice(range(150))
        if(array[index] == 1):
            array[index] = np.random.choice([2,3])
        elif(array[index] == 2):
            array[index] = np.random.choice([1,3])
        else:
            array[index] = np.random.choice([1,2])
    return array

#Function to calculate the training error and store all the training errors in a list

#Custom function to flip the values in train data and calulate decision boundry and training data prediction
def flip_values(flips,y_train,k,error_count):
    for flip in flips:
        error = 0
        y_train_change = y_train
        y_train_change = flip_function(y_train_change,flip)
        #print(np.unique(y_train_change, return_counts=True))
        decision = knn_predict(X12, X_train, y_train_change,k)
        decision = decision.reshape(x1.shape)
        plot(decision,y_train_change,flip)
        for i in range(len(X_train)):  
            error_count_flip = []   
            X_test_loocv = X_train[i].reshape(1,2)
            X_train_loocv = np.delete(X_train,i,axis=0)
            y_test_loocv = y_train_change[i].reshape(1,)
            y_train_loocv = np.delete(y_train_change,i,axis=0)
            #print(y_test_loocv.shape,y_train_loocv.shape)
            #print(X_test_loocv.shape,X_train_loocv.shape)
            decision_x_train_loocv = knn_predict(X_test_loocv,X_train_loocv,y_train_loocv,k)
            if(decision_x_train_loocv != y_test_loocv):
                error+= 1
        error_count.append(error)
    return y_train_loocv,error_count

#Function to plot the decision boundry based on the training data and test data
def plot(decision,y_train_change,flip):
    plt.figure()
    plt.pcolormesh(x1, x2, decision, cmap=cmap_light)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_change, cmap=cmap_bold, s=25)
    plt.title('Number of Flips = ' + str(flip))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.show()
    return

#Funtion to print the training error rates for each flips
def print_train_error_rate(flips,error_count,y_train_change):
    for flip,errors in zip(flips,error_count):
        print('Error Count with ' + str(flip) + ' flips is : ' + str(errors))
        print('Error Rate with ' + str(flip) + ' flips is : ' + str(np.round(errors/len(y_train_change)*100,3)))
        print('\n')
    return

#Main Function that calls flipping,plotting and error
def main():
    flips = [10,20,30,50]
    error_count = []
    k = 3
    y_train_loocv,error_count= flip_values(flips,y_train,k,error_count)
    #print(error_count)
    print_train_error_rate(flips,error_count,y_train_loocv)
    return

if __name__ == '__main__':
    main()
