import numpy as np
import ModelFunctions

# Hyperparameters of model
hiddenSize = 100
interHiddenSize = 500
learningRate = 0.0005
batchSize = 50
size = 100
lambdaVal = 0.0001
beta1 = 0.9  # Decay rate for the first moment estimates
beta2 = 0.999  # Decay rate for the second moment estimates
epsilon = 1e-8  # Small value to prevent division by zero

# Open the file, read the contents, and convert to lowercase
data = open('ScifiData.txt', 'r').read().lower()  # Convert text to lowercase

# Create a set of unique characters after converting to lowercase
chars = list(set(data))
# Create mappings from characters to indices and vice versa
charToIndex = {ch: i for i, ch in enumerate(chars)}
indexToChar = {i: ch for i, ch in enumerate(chars)}
# Calculate the size of the dataset and the vocabulary
dataSize, vocab = len(data), len(chars)

# Initialize weight matrices and bias vectors
U = np.random.uniform(low=-0.01, high=0.01, size=(vocab, hiddenSize))
V = np.random.uniform(low=-0.01, high=0.01, size=(hiddenSize, vocab))
W = np.random.uniform(low=-0.01, high=0.01, size=(hiddenSize, hiddenSize))
Q = np.random.uniform(low=-0.01, high=0.01, size=(hiddenSize, interHiddenSize))
R = np.random.uniform(low=-0.01, high=0.01, size=(interHiddenSize, interHiddenSize))
S = np.random.uniform(low=-0.01, high=0.01, size=(interHiddenSize, hiddenSize))

a = np.zeros((1, interHiddenSize))
b = np.zeros((1, hiddenSize))
c = np.zeros((1, vocab))
d = np.zeros((1, hiddenSize))

# Initialize weight and vector derivative memories for adagrad
firstMomentdOfA = np.zeros_like(a)
firstMomentdOfB = np.zeros_like(b)
firstMomentdOfC = np.zeros_like(c)
firstMomentdOfD = np.zeros_like(d)

firstMomentdOfU = np.zeros_like(U)
firstMomentdOfW = np.zeros_like(W)
firstMomentdOfV = np.zeros_like(V)
firstMomentdOfQ = np.zeros_like(Q)
firstMomentdOfR = np.zeros_like(R)
firstMomentdOfS = np.zeros_like(S)

secondMomentdOfA = np.zeros_like(a)
secondMomentdOfB = np.zeros_like(b)
secondMomentdOfC = np.zeros_like(c)
secondMomentdOfD = np.zeros_like(d)

secondMomentdOfU = np.zeros_like(U)
secondMomentdOfW = np.zeros_like(W)
secondMomentdOfV = np.zeros_like(V)
secondMomentdOfQ = np.zeros_like(Q)
secondMomentdOfR = np.zeros_like(R)
secondMomentdOfS = np.zeros_like(S)

ta = 0
tb = 0
tc = 0
td = 0

tU = 0
tW = 0
tV = 0
tQ = 0
tR = 0
tS = 0

# Get starting loss point
average_loss = -np.log(1.0 / vocab) * batchSize  # Initial loss at iteration 0
iteration_counter, data_pointer = 0, 0
lossOverItterations = []
while iteration_counter < 10000:
    # Keep track of current iteration and place in batch

    if data_pointer + batchSize + 1 >= len(data) or iteration_counter == 0:
        prevHidden = np.zeros((1, hiddenSize))
        prevInterHidden = np.zeros((1, interHiddenSize))
        data_pointer = 0

    # Create a batch of batchSize and equal sized target array to compare
    Batch = [charToIndex[char] for char in data[data_pointer:data_pointer + batchSize]]
    Targets = [charToIndex[char] for char in data[data_pointer + 1:data_pointer + batchSize + 1]]

    # Run batch through full forward/backward pass to get derivatives
    loss, dOfU, dOfW, dOfV, dOfQ, dOfR, dOfS, dOfA, dOfB, dOfC, dOfD, prevHidden, prevInterHidden = ModelFunctions.mainPass(Batch, Targets, prevHidden,prevInterHidden, U, W, V, Q, R, S, a, b, c, d, lambdaVal)

    # Average loss over batches and print it every 100 iterations
    average_loss = average_loss * 0.999 + loss * 0.001
    if iteration_counter % 10 == 0:
        print('Iteration ' + str(iteration_counter) + ' loss: ' + str(average_loss))
        seed = Batch[-1]
        returnString = ModelFunctions.sampleLanguage(chars, seed, prevHidden, prevInterHidden, size, U, W, V, Q, R, S, a, b, c, d)
        print(returnString)

    if iteration_counter % 100 == 0:
        lossOverItterations.append(average_loss)

    # Update the weights and bias with RMSprop method
    for parameter, gradient, firstMoment, secondMoment, t in zip(
            [U, W, V, Q, R, S, a, b, c, d],
            [dOfU, dOfW, dOfV, dOfQ, dOfR, dOfS, dOfA, dOfB, dOfC, dOfD],
            [firstMomentdOfU, firstMomentdOfW, firstMomentdOfV, firstMomentdOfQ, firstMomentdOfR, firstMomentdOfS, firstMomentdOfA, firstMomentdOfB, firstMomentdOfC, firstMomentdOfD],
            [secondMomentdOfU, secondMomentdOfW, secondMomentdOfV, secondMomentdOfQ, secondMomentdOfR, secondMomentdOfS, secondMomentdOfA, secondMomentdOfB, secondMomentdOfC, secondMomentdOfD],
            [tU, tW, tV, tQ, tR, tS, ta, tb, tc, td]  # Time step counter for each parameter, initialized to 0 and updated every time
    ):
        t += 1  # Increment the time step
        firstMoment = beta1 * firstMoment + (1 - beta1) * gradient  # Update biased first moment estimate
        secondMoment = beta2 * secondMoment + (1 - beta2) * (gradient ** 2)  # Update biased second moment estimate
        firstMomentCorrected = firstMoment / (1 - beta1 ** t)  # Correct bias in first moment
        secondMomentCorrected = secondMoment / (1 - beta2 ** t)  # Correct bias in second moment
        l1_grad = lambdaVal * np.sign(parameter)  # L1 regularization gradient
        parameter += -learningRate * (firstMomentCorrected / (np.sqrt(secondMomentCorrected) + epsilon))


    data_pointer += batchSize
    iteration_counter += 1

ModelFunctions.plot_error_set(lossOverItterations)