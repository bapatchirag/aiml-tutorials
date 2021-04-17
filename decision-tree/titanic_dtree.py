# Import test and training datasets for the Titanic dataset
def import_dataset():
    from pandas import read_csv
    test_set = read_csv('datasets/titanic_test.csv', sep=',', header=0)
    train_set = read_csv('datasets/titanic_train.csv', sep=',', header=0)

    print("Datasets imported")

    return test_set, train_set

# Classify feature values: Feature Engineering
def feature_engineering(test_set, train_set):
    full_set = [train_set, test_set]
    
    # Edit and add features
    for dataset in full_set:
        # Create binary feature has_cabin if traveller has cabin
        dataset["Has_Cabin"] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

        # Create feature FamilySize from SibSp and Parch
        dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

        # Replace NULLS in Embarked feature with a "S"
        dataset["Embarked"] = dataset["Embarked"].fillna("S")

        # Replace NULLS in Fare feature with the median of Fares in training data
        dataset["Fare"] = dataset["Fare"].fillna(train_set["Fare"].median())

        # Replace NULLS in Age feature with random value within one standard deviation of the mean on either side of the Ages in the dataset
        age_avg = dataset["Age"].mean()
        age_std = dataset["Age"].std()
        age_null_count = dataset["Age"].isnull().sum()
        from numpy import isnan, random
        dataset.loc[isnan(dataset["Age"]), "Age"] = random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset["Age"] = dataset["Age"].astype(int)

        # Get standard titles from name
        def get_title(name):
            from re import search
            title_search = search(" ([A-Za-z]+)\.", name)
            if title_search:
                return title_search.group(1)
            return ""
        dataset["Title"] = ""
        dataset["Title"] = dataset["Name"].apply(get_title)

        # Get all non-standard title from name
        dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], "Rare")
        dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
        dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
        dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

    # Map data to integer values
    for dataset in full_set:
        # Mapping sex
        dataset["Sex"] = dataset["Sex"].map({"female": 0, "male": 1}).astype(int)

        # Mapping titles
        dataset["Title"] = dataset["Title"].map({"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}).fillna(0)

        # Mapping embarked
        dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

        # Mapping Fare
        dataset.loc[(dataset["Fare"] <= 7.91), "Fare"] = 0
        dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
        dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
        dataset.loc[(dataset["Fare"] > 31), "Fare"] = 3
        dataset["Fare"] = dataset["Fare"].astype(int)

        # Mapping Age
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
        dataset["Age"] = dataset["Age"].astype(int)

    # Remove unnecessary features now
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    for dataset in full_set:
        dataset = dataset.drop(drop_elements, axis=1)

    return test_set, train_set

# Get test and train x and y values
def get_test_train(train_set):
    X = train_set[["Pclass", "Sex", "Age", "Parch", "Fare", "Embarked", "Has_Cabin", "FamilySize", "Title"]]
    Y = train_set[["Survived"]]
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    return X_test, X_train, Y_test, Y_train

# Train dataset
def train_dataset(X_train, Y_train):
    from sklearn.tree import DecisionTreeClassifier
    trained_model = DecisionTreeClassifier(criterion="entropy", random_state=42)

    trained_model.fit(X_train, Y_train)

    return trained_model

# Test from test dataset
def test_dataset(X_test, trained_model):
    Y_prediction = trained_model.predict(X_test)
    return Y_prediction

# Get accuracy
def get_accuracy(Y_prediction, Y_test):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_test, Y_prediction)

    return accuracy

# Print confusion matrix
def get_confusion_matrix(Y_prediction, Y_test):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(Y_test, Y_prediction)

# Predict survivors
def predict_survivors(X_to_predict, trained_model):
    return test_dataset(X_to_predict, trained_model)

# Display decision tree
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

# Main script starts here
def main():
    # Train dataset and test against test data
    test_set, train_set = import_dataset()
    test_set, train_set = feature_engineering(test_set, train_set)
    X_test, X_train, Y_test, Y_train = get_test_train(train_set)
    trained_model = train_dataset(X_train, Y_train)
    Y_prediction = test_dataset(X_test, trained_model)

    # Get and print accuracy of test data
    accuracy = get_accuracy(Y_prediction, Y_test)
    print("Accuracy:", (accuracy*100))
    print("-"*50)

    # Get and print confusion matrix - Matrix of TPs, TNs, FPs and FNs
    print("Confusion matrix: ")
    print(get_confusion_matrix(Y_prediction, Y_test))
    print("-"*50)

    # Predict surviors in unlabeled dataset
    X_to_predict = test_set[["Pclass", "Sex", "Age", "Parch", "Fare", "Embarked", "Has_Cabin", "FamilySize", "Title"]]
    Y_predicted_np = predict_survivors(X_to_predict, trained_model)

    # Display survivors in readable format
    from pandas import DataFrame, concat
    Y_predicted_df = DataFrame(data=Y_predicted_np, columns=["Survived"])
    complete_predicted_data = Y_predicted_df.join(test_set[["PassengerId", "Name"]])
    print(complete_predicted_data)
    print("-"*50)

    # Display tree generated from given features and generated classes
    features = ["Pclass", "Sex", "Age", "Parch", "Fare", "Embarked", "Has_Cabin", "FamilySize", "Title"]
    classes = ["Died", "Survived"]
    show_tree(trained_model, features, classes)

if __name__ == "__main__":
    main()