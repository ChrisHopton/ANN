import numpy as np

# X = (hours sleeping, hours studying), y = test score of student


''' 
    This function was found on stack overflow -- Modified to fit my needs
    This function goes through a list of increasing hidden layers and iterations finding the combination
    that gets the best accuracy.
'''

def find_best_hyperparameters(x, y, hidden_sizes, num_iterations):
    best_loss = float('inf')
    best_hidden_size = 0
    best_num_iterations = 0

    for hidden_size in hidden_sizes:
        for num_iteration in num_iterations:
            NN = NeuralNetwork(hidden_size=hidden_size)
            for i in range(num_iteration):
                NN.train(x, y)
            loss = np.mean(np.square(y - NN.feedForward(x)))
            if loss < best_loss:
                best_loss = loss
                best_hidden_size = hidden_size
                best_num_iterations = num_iteration

    return best_hidden_size, best_num_iterations

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def k_fold_cross_validation(X, y, k=5, hidden_size=3, num_iterations=500):
    indices = list(range(len(X)))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    fold_accuracies = []

    for fold in range(k):
        test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
        train_indices = [i for i in indices if i not in test_indices]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        NN = NeuralNetwork(hidden_size=hidden_size)

        for i in range(num_iterations):
            NN.train(X_train, y_train)

        y_pred = np.round(NN.feedForward(X_test))
        fold_accuracy = accuracy(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} accuracy: {fold_accuracy:.2f}")

    mean_accuracy = np.mean(fold_accuracies)
    print(f"Total average accuracy of all folds: {mean_accuracy:.2f}")
    return mean_accuracy


class NeuralNetwork(object):
    def __init__(self, hidden_size=3,weight_decay=0.1):
        self.inputSize = 2
        self.hiddenSize = hidden_size
        self.outputSize = 1
        self.weight_decay = weight_decay

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) * np.sqrt(1 / self.inputSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) * np.sqrt(1 / self.hiddenSize)

        self.learning_rate = .01
        


    def feedForward(self, x):
        # Forward propagation through the network
        self.z = np.dot(x, self.W1) # dot product of x (input matrix) and first set of weights --> This is the first layer
        self.z2 = self.sigmoid(self.z) # Pass the first layer --> activation function ( 2nd layer)
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of weights
        output = self.sigmoid(self.z3) # feed final hidden layer
        return output

    def sigmoid(self, activation, deriv=False):
        if(deriv):
            return activation * (1-activation)
        return 1/(1+np.exp(-activation))
    
    def backward(self, x, y, output):
        # Backward propagate through the network
        self.output_error = y - output # Desired output - our actual output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True) # Gradient of the error

        self.z2_error = self.output_delta.dot(self.W2.T) # Find error in hidden layer: How much our hidden layer weight contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) # Apply deriv of sigmoid to hidden layer z2 error

        # Update weights with weight decay (L2 regularization)
        self.W1 += self.learning_rate * (x.T.dot(self.z2_delta) - self.weight_decay * self.W1) # adjust first set (input -> hidden) weights
        self.W2 += self.learning_rate * (self.z2.T.dot(self.output_delta) - self.weight_decay * self.W2) # adjust second set (hidden -> output) weights


    def train(self, x, y):
        output = self.feedForward(x)
        self.backward(x, y, output)



filename = "C:\\Users\\mathe\\Dropbox\\PC\Desktop\\AI\\Project 1\\labeled-examples"
# Read and parse the input data
with open(filename, "r") as file:
    lines = file.readlines()
    data = [line.split() for line in lines]

#Use this x for Labeled-Examples
x = np.array([[float(row[1]), float(row[2])] for row in data])
y = np.array([[int(row[0])] for row in data])

# Use this x and yfor acc
# x = np.array([[float(row[1]), float(row[2]),float(row[3]),float(row[4])] for row in data])
# y = (y - min(y)) / (max(y) - min(y))

#Use this x and y for seeds
# x = np.array([[float(row[0]), float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6])] for row in data])
# y = np.array([[int(row[7])] for row in data])
# y = (y - min(y)) / (max(y) - min(y))

x = x / np.amax(x, axis=0)
y = y.astype(float)

# Define the grid search space
hidden_sizes = [2, 3, 4, 5,10,50]
num_iterations = [500, 1000, 2000, 5000]

# Find the best hyperparameters
hiddenSize, iterationSize = find_best_hyperparameters(x, y, hidden_sizes, num_iterations)
print("Best Hidden Size: ", hiddenSize, " Best Iterations: " , iterationSize)

# hiddenSize = 5
# iterationSize = 1000

# Train and evaluate the neural network using k-fold cross-validation
mean_accuracy = k_fold_cross_validation(x, y, k=5, hidden_size=hiddenSize, num_iterations=iterationSize)



