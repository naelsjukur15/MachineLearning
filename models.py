import nn
import numpy as np
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)
        scoreToScalar = nn.as_scalar(score)
        if scoreToScalar >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            mistakeMade = False
            for inputData, output in dataset.iterate_once(1):
                currentPrediction = self.get_prediction(inputData)
                if nn.as_scalar(output) != currentPrediction:
                    self.w.update(inputData, nn.as_scalar(output))
                    mistakeMade = True
            if not mistakeMade:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.W1 = nn.Parameter(1, 512)
        self.b1 = nn.Parameter(1, 512)
        self.W2 = nn.Parameter(512, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        xw1 = nn.Linear(x, self.W1)
        hiddenLayer = nn.ReLU(nn.AddBias(xw1, self.b1))
        xw2 = nn.Linear(hiddenLayer, self.W2)
        toReturn = nn.AddBias(xw2, self.b2)
        return toReturn



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(x)
        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        patienceCounter = 5
        slowImprovement = 0
        learningRateDecay = 0.9
        learningRate = 0.05
        batchSize = 200
        totalEpoch = 0
        minimumLoss = float('inf')


        while True:

            total_loss = 0
            num_batches = 0

            for i, j in dataset.iterate_once(batchSize):
                loss = self.get_loss(i, j)
                total_loss += nn.as_scalar(loss)
                num_batches += 1

                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(gradients[0], -learningRate)
                self.b1.update(gradients[1], -learningRate)
                self.W2.update(gradients[2], -learningRate)
                self.b2.update(gradients[3], -learningRate)

            average_loss = total_loss / num_batches

            if average_loss < minimumLoss:
                minimumLoss = average_loss
                slowImprovement = 0
            else:
                slowImprovement += 1
                if slowImprovement >= patienceCounter:
                    learningRate *= learningRateDecay
                    slowImprovement = 0

            if average_loss < 0.02:
                break
            totalEpoch += 1

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.W1 = nn.Parameter(784, 200)  # First layer weights
        self.b1 = nn.Parameter(1, 200)  # First layer bias
        self.W2 = nn.Parameter(200, 10)  # Second layer weights
        self.b2 = nn.Parameter(1, 10)  # Second layer bias

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        xw1 = nn.Linear(x, self.W1)
        hiddenLayer = nn.ReLU(nn.AddBias(xw1, self.b1))
        xw2 = nn.Linear(hiddenLayer, self.W2)
        toReturn = nn.AddBias(xw2, self.b2)

        return toReturn

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learningRate = 0.5
        batchSize = 100
        accuracyThreshold = 0.98

        while True:
            for x, y in dataset.iterate_once(batchSize):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(gradients[0], -learningRate)
                self.b1.update(gradients[1], -learningRate)
                self.W2.update(gradients[2], -learningRate)
                self.b2.update(gradients[3], -learningRate)

            validationAccuracy = dataset.get_validation_accuracy()
            if validationAccuracy >= accuracyThreshold:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.batch_size = 100
        self.learning_rate = 0.5

        # Initialize your model parameters here
        self.languageCount = 5
        self.hiddenLayerSize = 100
        self.W_initial = nn.Parameter(self.num_chars, self.hiddenLayerSize)
        self.W_hidden = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.b_hidden = nn.Parameter(1, self.hiddenLayerSize)
        self.W_output = nn.Parameter(self.hiddenLayerSize, self.languageCount)
        self.b_output = nn.Parameter(1, self.languageCount)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        hidden = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W_initial), self.b_hidden))
        for x in xs[1:]:
            hidden = nn.ReLU(nn.Add(nn.Linear(x, self.W_initial),
                               nn.AddBias(nn.Linear(hidden, self.W_hidden), self.b_hidden)))

        scores = nn.AddBias(nn.Linear(hidden, self.W_output), self.b_output)
        return scores

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        scores = self.run(xs)
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        maxGradient= 5
        earlyStopPatience = 3
        noImprovementEpoch = 0
        validationAccuracyBest = 0
        initialLearningRate = 0.1
        learningRateDecay = 0.9
        presentLearningRate = initialLearningRate

        while True:
            totalLoss = 0
            batchesCount = 0

            for i, j in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(i, j)
                totalLoss += nn.as_scalar(loss)
                batchesCount += 1


                gradients = nn.gradients(loss,
                                         [self.W_initial, self.W_hidden, self.b_hidden, self.W_output, self.b_output])
                gradients = [nn.Constant(np.clip(g.data, -maxGradient, maxGradient)) for g in gradients]

                self.W_initial.update(gradients[0], -presentLearningRate)
                self.W_hidden.update(gradients[1], -presentLearningRate)
                self.b_hidden.update(gradients[2], -presentLearningRate)
                self.W_output.update(gradients[3], -presentLearningRate)
                self.b_output.update(gradients[4], -presentLearningRate)

            currValidationAccuracy = dataset.get_validation_accuracy()

            if currValidationAccuracy > validationAccuracyBest:
                validationAccuracyBest = currValidationAccuracy
                noImprovementEpoch = 0
            else:
                noImprovementEpoch += 1
                if noImprovementEpoch >= earlyStopPatience:
                    break

            presentLearningRate *= learningRateDecay