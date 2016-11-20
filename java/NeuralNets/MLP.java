package NeuralNets;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
import java.util.Arrays;
import java.util.Random;

public class MLP {

  public double[][][] weights;  /*
    `weights[l][i][j]` is the weight of the connection in level l between neuron i
    in layer l and neuron j in layer l+1.

    Synaptic connections organized into "levels" which are distinct from
    "layers": a layer is a collection of neurons in the same strata of the MLP,
    while a level is a collection of connections between neurons in two distinct
    layers. An MLP with L layers has L-1 levels.
  */

  public double[][] bias;  /*
    `bias[l][i]` is the bias of the output neuron i in level l, i.e. the bias of
    the i-th neuron in the (l+1)-th layer.
  */

  private int[] layers;
  private double error;
  private String[] transferFunctions;
  private double[][] deltas;
  private double[][] neuronOutputs;  /*
    Private variables used by various methods.
    `transferFunctions[l]` and `deltas[l][i]` are for output neurons in level l,
    i.e. neurons in layer l+1.
    `neuronOutputs[l][i]` is the output of neuron i in *layer* l.
  */

  public MLP(int[] layers, String[] transferFunctions, double lower, double upper, long seed) {  /*
    Construct a multilayer perceptron.

    Parameters:
    layers - i-th entry gives the number of neurons on layer i.
    transferFunctionTypes - Entry i should be one of the strings "tanh", "logistic"
      or "identity" specifying the transfer function to be used by all the neurons in layer i.
      Defaults to "identity" if an unrecognized option is passed.
    lower, upper - Lower and upper bounds for the randomly initialized synaptic weights.
    seed - Seed to the random number generator used to initialize the synaptic weights.
    */

    this.layers = layers;
    this.transferFunctions = transferFunctions;

    int numLayers = layers.length;
    int numLevels = numLayers - 1;

    this.weights = new double[numLevels][][];
    this.bias = new double[numLevels][];
    this.deltas = new double[numLevels][];
    this.neuronOutputs = new double[layers.length][];

    Random rng = new Random(seed);
    for (int l = 0; l < numLevels; l++) {
      int n = layers[l];
      int m = layers[l+1];
      this.weights[l] = new double[n][m];
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          this.weights[l][i][j] = (upper - lower) * (rng.nextDouble() - 0.5);
        }
      }
      this.bias[l] = new double[m];
      for (int i = 0; i < m; i++) {
        this.bias[l][i] = (upper - lower) * (rng.nextDouble() - 0.5);
      }
      this.deltas[l] = new double[m];
      this.neuronOutputs[l] = new double[n];
    }
    this.neuronOutputs[numLayers - 1] = new double[layers[numLayers - 1]];
  }

  static private double logistic(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  public void train(double[] learningRates, double[][] inputVectors, double[][] outputVectors, long steps, long seed, boolean outputLearningCurve) {  /*
    Use backpropagation to train the MLP, by drawing randomly and uniformly with
    replacement from a given dataset and updating the weights after every training
    instance.

    Parameters:
    learningRates - Learning rates of each level of the network.
    inputVectors, outputVectors - Training input and corresponding output.
    steps - Number of samples to draw.
    seed - Seed to the random number generator used to draw the samples.
    outputLearningCurve - If true, output learning curve error data to file "learning.curve". File is plottable with gnuplot.
    */

    BufferedWriter output = null;
    if (outputLearningCurve) {
      try {
        FileWriter outputFile = new FileWriter("learning.curve");
        output = new BufferedWriter(outputFile);
        output.write("# Column 1: Training step, Column 2: Error\n");
      } catch (IOException e) {
        System.out.println("Couldn't create output learning curve file!\n");
      }
    }

    double[][][] weightChanges;
    int data;
    Random rng = new Random(seed);
    for (int step = 0; step < steps; step++) {
      data = rng.nextInt(inputVectors.length);
      weightChanges = backpropagate(learningRates, inputVectors[data], outputVectors[data]);
      for (int l = 0; l < this.weights.length; l++) {
        for (int i = 0; i < this.weights[l].length; i++) {
          for (int j = 0; j < this.weights[l][i].length; j++) {
            this.weights[l][i][j] += weightChanges[l][i][j];
          }
        }
      }

      if (outputLearningCurve && output != null) {
        try{
          output.write(step + " " + this.error + "\n");
        } catch (IOException e) {
          System.out.println("Problem writing to file.\n");
        }
      }
    }

    if (outputLearningCurve && output != null) {
      try {
        output.close();
      } catch (IOException e) {
        System.out.println("Couldn't close file \"learning.curve\".\n");
      }
    }
  }

  private double[][][] backpropagate(double[] learningRates, double[] X, double[] Y) {  /*
    Return the weight changes obtained via backpropagation with the single training
    input X and training output Y.
    Also set the error between the training and actual output in `this.error`.

    `learningRates[l]` is the learning rate for the output neurons in level l.
    */

    int numLevels = this.layers.length - 1;

    double[][][] weightChanges = new double[numLevels][][];

    double[] outputY = this._feed(X, 0, true);  /*
      Pass the input through the network, calculating and storing the output of
      each neuron.
    */

    this.error = 0;
    for (int i = 0; i < Y.length; i++) {
      this.error += Math.pow(Y[i] - outputY[i], 2);
    }

    // Backpropagate to calculate deltas and weight changes for all other levels.
    for (int level = numLevels - 1; level >= 0; level--) {
      int inputLayer = level;
      int outputLayer = inputLayer + 1;
      int n = this.layers[inputLayer];
      int m = this.layers[outputLayer];
      weightChanges[level] = new double[n][m];
      for (int j = 0; j < m; j++) {
        double out = this.neuronOutputs[outputLayer][j];
        double transferDerivative = (this.transferFunctions[level] == "tanh") ? 1.0 - Math.pow(out, 2) : (this.transferFunctions[level] == "logistic") ? out * (1 - out) : 1;

        if (level < numLevels - 1) {  // Hidden neurons.
          double weightedDeltaSum = 0;
          for (int k = 0; k < this.layers[outputLayer + 1]; k++) {
            weightedDeltaSum += this.deltas[level + 1][k] * this.weights[level + 1][j][k];
          }
          this.deltas[level][j] = weightedDeltaSum * transferDerivative;
        } else {  // Output neurons.
          this.deltas[level][j] = (Y[j] - out) * transferDerivative;
        }

        for (int i = 0; i < n; i++) {
          weightChanges[level][i][j] = learningRates[level] * this.deltas[level][j] * this.neuronOutputs[inputLayer][i];
        }
      }
    }

    return weightChanges;
  }

  private double[] _feed(double[] X, int level, boolean storeNeuronOutputs) {  /*
    !!! Not meant for public consumption !!!

    Recursively pass the input vector X through the network beginning at level
    `level`.
    If `storeNeuronOutputs == true` then store the output values of each neuron
    in `this.neuronOutputs`.
    */
    if (level >= this.weights.length) {
      return X;
    } else {
      int inputLayer = level;
      int outputLayer = inputLayer + 1;
      int inputDim = this.layers[inputLayer];
      int outputDim = this.layers[outputLayer];
      double[] Y = new double[outputDim];
      for (int j = 0; j < outputDim; j++) {
        double y = this.bias[level][j];
        for (int i = 0; i < inputDim; i++) {
          y += X[i] * this.weights[level][i][j];
        }
        Y[j] = (this.transferFunctions[level] == "tanh") ? Math.tanh(y) : (this.transferFunctions[level] == "logistic") ? logistic(y) : y;
        if (storeNeuronOutputs) {
          this.neuronOutputs[outputLayer][j] = Y[j];
          if (level == 0) {
            this.neuronOutputs[0] = X;
          }
        }
      }
      return _feed(Y, level + 1, storeNeuronOutputs);
    }
  }

  public double[] feed(double[] X) {  /*
    Return the result of passing the input vector X through the network.
    */
    return _feed(X, 0, false);
  }

  public String toString() {
    return "MLP(" + Arrays.deepToString(this.weights) + ", " + Arrays.deepToString(this.bias) + ")";
  }
}
