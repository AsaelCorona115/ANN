from asyncio.windows_events import NULL
import math
# Global Values
alpha = 0.1

# Input for training
x1 = 0
x2 = 0
expectedOutput = 0

# The node class takes the inputs, weights as arrays. It has built in dummydata
# and dummyWeight attributes. This class calculates Inj and g(Inj) and assigns them to
# the node as attributes.
# When the object is created the input and weight values must be in order.
# For instance: the second weight corresponds to the weight of the connection
# between input two to the node.


class Node:
    def __init__(self, inputs, weights):
        self.dummyInput = 1
        self.dummyWeight = 0.1
        self.inputs = inputs
        self.weights = weights
        self.Inj = self.calculateInj()
        self.a = self.calculateA()

    def calculateInj(self):
        InputWeightSum = 0
        for x in range(len(self.inputs)):
            InputWeight = (self.inputs[x]) * (self.weights[x])
            InputWeightSum = InputWeightSum + InputWeight
        Inj = InputWeightSum + self.dummyWeight
        return round(Inj, 3)

    def calculateA(self):
        # Calculating the output of the node
        output = round(1 / (1 + (math.exp(self.Inj * -1))), 3)
        return output


# Creating the nodes
node3 = Node([x1, x2], [0.1, 0.1])
node4 = Node([x1, x2], [0.1, 0.1])
node5 = Node([node3.a, node4.a], [0.1, 0.1])


nodes = [node3, node4, node5]
# ***************************This concludes initialization*************************

# This function calculates the delta value for the output node


def outputDelta(predictedValue, TrueValue):
    gPrime = predictedValue * (1 - predictedValue)
    delta = round(((TrueValue - predictedValue) * gPrime), 3)
    return delta

# This function calculates the delta value of the hidden layer nodes


def hiddenDelta(aJ, OutputWeight, Outputdelta):
    gPrime = aJ * (1 - aJ)
    delta = round(gPrime * OutputWeight * Outputdelta, 3)
    return delta

# This function calculates the new weight after delta value has been calculated.


def weightUpdate(originalWeight, learningRate, inputValue, deltaValue):
    newWeight = round(
        originalWeight + (learningRate * inputValue * deltaValue), 4)
    return newWeight


def updatingWeights(listOfNodes):
    # Calculating the delta values for output node:
    delta5 = outputDelta(node5.a, expectedOutput)
    delta4 = hiddenDelta(node4.a, node5.weights[1], delta5)
    delta3 = hiddenDelta(node3.a, node5.weights[0], delta5)
    deltas = [delta3, delta4, delta5]

    for x in range(len(listOfNodes)):
        listOfNodes[x].dummyWeight = weightUpdate(
            listOfNodes[x].dummyWeight, alpha, listOfNodes[x].dummyInput, deltas[x])
        listOfNodes[x].weights[0] = weightUpdate(
            listOfNodes[x].weights[0], alpha, listOfNodes[x].inputs[0], deltas[x])
        listOfNodes[x].weights[1] = weightUpdate(
            listOfNodes[x].weights[1], alpha, listOfNodes[x].inputs[1], deltas[x])


updatingWeights(nodes)
print(node5.a)
