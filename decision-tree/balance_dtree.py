import pandas as pd
import numpy as np

# Get balance scale dataset from CSV
def import_dataset():
    balance = pd.read_csv('datasets/balance-scale.csv', sep=',', header=None)
    print("Balance scale dataset imported")
    return balance

# Split dataset into testing and training dataset in a 30%-70% split
def get_test_train(balance):
    X = balance.values[:, 1:5]
    Y = balance.values[:, 0]

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    return X_train, X_test, Y_train, Y_test

# Train dataset using the values in the training set
def train_dataset(X_train, Y_train):
    from sklearn.tree import DecisionTreeClassifier
    trained_model = DecisionTreeClassifier(criterion="entropy", random_state=42)

    trained_model.fit(X_train, Y_train)

    return trained_model

# Test from testing set
def test_dataset(X_test, trained_model):
    Y_prediction = trained_model.predict(X_test)
    return Y_prediction

# Get accuracy of the model derived
def get_accuracy(Y_prediction, Y_test):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_test, Y_prediction)

    return accuracy

# Print confusion matrix for given prediction and test data
def get_confusion_matrix(Y_prediction, Y_test):
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, Y_prediction))

# Main script starts here
def main():
    balance = import_dataset()
    X_train, X_test, Y_train, Y_test = get_test_train(balance)
    trained_model = train_dataset(X_train, Y_train)
    Y_prediction = test_dataset(X_test, trained_model)
    accuracy = get_accuracy(Y_prediction, Y_test)
    print("Accuracy:", accuracy)
    get_confusion_matrix(Y_prediction, Y_test)

if __name__ == "__main__":
    main()