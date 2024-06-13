
#James Panagis 


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, max_iterations=100):
    #perceptron with given parameters
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    #weights and bias    
        self.weights = np.zeros(num_features)
        self.bias = 0

    def activation(self, x):
    #function returns 1 if x > 0 else 0
        return 1 if x > 0 else 0

    def predict(self, x):
    #predicts class label based on input features and current weight
        return self.activation(np.dot(self.weights, x) + self.bias)

    def train(self, X_train, y_train):
    # training the perceptron model
        for _ in range(self.max_iterations):
            for i in range(len(X_train)):
                prediction = self.predict(X_train[i])
                error = y_train[i] - prediction
                # Update weights and bias
                self.weights += self.learning_rate * error * X_train[i]
                self.bias += self.learning_rate * error

#function to evaluate model performance
def evaluate_performance(model, X_test, y_test):
    y_pred = [model.predict(x) for x in X_test]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# data from excel. I coudnt set a file path to directory because the program couldnt recognize it. It might be my harddrive
X = np.array([
    [1, 0.08, 0.72],
    [1, 0.1, 1],
    [1, 0.26, 0.58],
    [1, 0.35, 0.95],
    [1, 0.45, 0.15],
    [1, 0.6, 0.3],
    [1, 0.7, 0.65],
    [1, 0.92, 0.45],
    [1, 0.42, 0.85],
    [1, 0.65, 0.55],
    [1, 0.2, 0.3],
    [1, 0.2, 1],
    [1, 0.85, 0.1]
])
#created an array that holds that data to train
y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1])

#spliting the data into training and testing sets 70% training 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#perceptron model
perceptron = Perceptron(num_features=X_train.shape[1])

#list to store evaluation results for different iterations
evaluation_results = []

#numbering each model performance with different numbers of training iterations
training_iterations = [1, 5, 10, 20, 30, 40, 50]
for num_iterations in training_iterations:
    #traiing model
    perceptron.max_iterations = num_iterations
    perceptron.train(X_train, y_train)
    
    #test performance on dataset
    accuracy, precision, recall, f1 = evaluate_performance(perceptron, X_test, y_test)
    evaluation_results.append((num_iterations, accuracy, precision, recall, f1))

#print results
print("Evaluation results:")
print("Iterations\tAccuracy\tPrecision\tRecall\t\tF1-score")
for result in evaluation_results:
    print("{}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(*result))
