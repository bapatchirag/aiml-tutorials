# Get balance scale dataset from CSV
def import_dataset():
    from pandas import read_csv
    balance = read_csv('datasets/balance-scale.csv', sep=',', header=None)
    print("Balance scale dataset imported")
    return balance

# Split dataset into testing and training dataset in a 30%-70% split
def get_test_train(balance):
    X = balance.values[:, 1:5]
    Y = balance.values[:, 0]

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    return X_train, X_test, Y_train, Y_test

# Train dataset and get trained model
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
def print_confusion_matrix(Y_prediction, Y_test):
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, Y_prediction))

# Display tree
def show_tree(trained_model, features, classes):
    with open("model.dot", "w") as f:
        from sklearn.tree import export_graphviz
        f = export_graphviz(trained_model, out_file=f, impurity=True, feature_names=features, class_names=classes, rounded=True, filled=True)

    # Convert .dot to .png
    from subprocess import check_call
    check_call(["dot", "-Tpng", "model.dot", "-o", "model.png"])
    
    # Open tree from command line
    from platform import system
    os_name = system()
    if os_name == "Linux":
        check_call(["xdg-open", "model.png"])
    elif os_name == "Windows":
        check_call(["model.png"])

# Main script begins here
def main():
    # Train dataset and test against test data
    balance = import_dataset()
    X_train, X_test, Y_train, Y_test = get_test_train(balance)
    trained_model = train_dataset(X_train, Y_train)
    Y_prediction = test_dataset(X_test, trained_model)

    # Get and print accuracy of test data
    accuracy = get_accuracy(Y_prediction, Y_test)
    print("Accuracy:", (accuracy*100))
    print("-"*20)

    # Get and print confusion matrix - Matrix of TPs, TNs, FPs and FNs
    print("Confusion matrix:")
    print_confusion_matrix(Y_prediction, Y_test)
    print("-"*20)

    # Display tree generated from given features and generated classes
    features = ["left-weight", "left-distance", "right-weight", "right-distance"]
    classes = ["B", "L", "R"]
    show_tree(trained_model, features, classes)

if __name__ == "__main__":
    main()