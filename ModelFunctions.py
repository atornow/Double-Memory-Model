import numpy as np
import matplotlib.pyplot as plt

def hiddenFunction(a):
    # Compute tanh of input vector, add non-linearity.
    return np.tanh(a)

def softMax(x):
    # Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def derivativeLofO(Actual, Desired):
    # Ensure Actual and Desired are NumPy arrays to use NumPy operations
    Actual = np.array(Actual)
    Desired = np.array(Desired)

    # Compute the derivative
    dL_dP = Actual - Desired

    return dL_dP

def crossEntropy(Actual, Desired):
    # Ensure Actual and Desired are NumPy arrays to use NumPy operations
    Actual = np.array(Actual)
    Desired = np.array(Desired)
    # Calculate the cross entropy loss for given array compared to desired array
    loss = -np.sum(Desired * np.log(Actual))
    return loss


def mainPass(Batch, Targets, prevHidden, prevInterHidden, U, W, V, Q, R, S, a,  b, c, d, lambdaVal):

    oneHot = {}
    savedHiddens = {}
    savedInterHiddens = {}
    savedLogits = {}
    savedInterLogits = {}
    savedProbs = {}
    savedHiddens[-1] = np.copy(prevHidden)
    savedInterHiddens[-1] = np.copy(prevInterHidden)
    loss = 0

    # Forward pass through Batch
    for char in range(len(Batch)):

        oneHot[char] = np.zeros_like(c)  # encode in 1-of-vocab representation
        oneHot[char][0, Batch[char]] = 1

        savedHiddens[char] = hiddenFunction(b + np.dot(savedHiddens[char - 1], W) + np.dot(oneHot[char], U))
        savedInterHiddens[char] = hiddenFunction(a + np.dot(savedInterHiddens[char - 1], R) + np.dot(savedHiddens[char], Q))

        savedInterLogits[char] = d + np.dot(savedInterHiddens[char], S)
        savedLogits[char] = c + np.dot(savedInterLogits[char], V)
        savedProbs[char] = softMax(savedLogits[char])

        prob_array_for_char = savedProbs[char]
        target_prob = prob_array_for_char[0][Targets[char]]
        loss += -np.log(target_prob)

    # Calculate L1 penalty for weight matrices
    l1_penalty = lambdaVal * (np.sum(np.abs(U)) + np.sum(np.abs(W)) + np.sum(np.abs(V)))
    # Add L1 penalty to the loss
    loss += l1_penalty

    dOfA = np.zeros_like(a)
    dOfB = np.zeros_like(b)
    dOfC = np.zeros_like(c)
    dOfD = np.zeros_like(d)
    dOfU = np.zeros_like(U)
    dOfW = np.zeros_like(W)
    dOfV = np.zeros_like(V)
    dOfQ = np.zeros_like(Q)
    dOfR = np.zeros_like(R)
    dOfS = np.zeros_like(S)

    dNextHidden = np.zeros_like(savedHiddens[0])
    dNextInterHidden = np.zeros_like(savedInterHiddens[0])

    # Backwards pass through sequence
    for char in reversed(range(len(Batch))):

        # Helper variables for getting derivatives over batch
        dLogit = np.copy(savedProbs[char])
        dLogit[0][Targets[char]] -= 1
        dInterLogit = np.dot(dLogit, np.transpose(V))

        dOfI = dNextInterHidden + np.dot(dInterLogit, np.transpose(S))
        dOfIRaw = np.dot(dOfI, np.diag((1 - (np.square(savedInterHiddens[char]))).flatten()))

        dOfH = dNextHidden + np.dot(dOfIRaw, np.transpose(Q))
        dOfHRaw = np.dot(dOfH, np.diag((1 - (np.square(savedHiddens[char]))).flatten()))

        # Calculate and sum derivatives over batch
        dOfB += dOfHRaw
        dOfC += dLogit
        dOfU += np.dot(np.transpose(oneHot[char]), dOfHRaw)
        dOfW += np.dot(np.transpose(savedHiddens[char - 1]), dOfHRaw)
        dOfV += np.dot(np.transpose(savedInterLogits[char]), dLogit)
        dOfS += np.dot(np.transpose(savedInterHiddens[char]), dInterLogit)
        dOfD += dInterLogit
        dOfR += np.dot(np.transpose(savedInterHiddens[char - 1]), dOfIRaw)
        dOfQ += np.dot(np.transpose(savedHiddens[char]), dOfIRaw)
        dOfA += dOfIRaw

        # Update change in hidden with respect to previous hidden
        dNextHidden = np.dot(dOfHRaw, W.T)
        dNextInterHidden = np.dot(dOfIRaw, R.T)

    # Clip values in derivatives to prevent explosions in gradient
    for dParam in [dOfU, dOfW, dOfV, dOfQ, dOfR, dOfS, dOfA, dOfB, dOfC, dOfD]:
        np.clip(dParam, -5, 5, out=dParam)

    # Return summed loss and derivatives aswell as the last state vector for next pass
    return loss, dOfU, dOfW, dOfV, dOfQ, dOfR, dOfS, dOfA, dOfB, dOfC, dOfD, savedHiddens[len(Batch) - 1], savedInterHiddens[len(Batch) - 1]

def sampleLanguage(chars, seed, prevHidden, prevInterHidden, size, U, W, V, Q, R, S, a, b, c, d):
    indexToChar = {i: ch for i, ch in enumerate(chars)}
    oneHot = {}
    savedHiddens = {}
    savedInterHiddens = {}
    savedLogits = {}
    savedInterLogits = {}
    savedProbs = {}
    savedHiddens[-1] = np.copy(prevHidden)
    savedInterHiddens[-1] = np.copy(prevInterHidden)

    returnString = indexToChar[seed]

    # Forward pass through Batch
    for char in range(size):
        oneHot[char] = np.zeros_like(c)  # encode in 1-of-vocab representation

        if char == 0:
            oneHot[char][0, seed] = 1
        else:
            oneHot[char][0, nextChar] = 1

        savedHiddens[char] = hiddenFunction(b + np.dot(savedHiddens[char - 1], W) + np.dot(oneHot[char], U))
        savedInterHiddens[char] = hiddenFunction(a + np.dot(savedInterHiddens[char - 1], R) + np.dot(savedHiddens[char], Q))

        savedInterLogits[char] = d + np.dot(savedInterHiddens[char], S)
        savedLogits[char] = c + np.dot(savedInterLogits[char], V)
        savedProbs[char] = softMax(savedLogits[char])

        prob_array_for_char = savedProbs[char]
        nextChar = np.argmax(prob_array_for_char)
        returnString += (indexToChar[np.argmax(prob_array_for_char)])

    return returnString

def plot_error_set(ErrorSet):
    # Determine the size of ErrorSet
    size = len(ErrorSet)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(ErrorSet, label='Error')

    # Scaling the x-axis based on the size of ErrorSet
    if size > 10000000:
        ax.set_xscale('log')
        ax.set_xlabel('Epoch (log scale)')
    else:
        ax.set_xlabel('Epoch')

    # Enhancements for readability
    ax.set_ylabel('Error')
    ax.set_title('Averaged error over all epochs')
    ax.grid(True)
    ax.legend()

    # Show the plot
    plt.show()
