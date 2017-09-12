package cnn;

import data.Category;
import data.DataSet;
import data.Instance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static cnn.ActivationFunction.SIGMOID;
import static data.Category.categories;
import static data.Category.categoryToOneOfN;
import static driver.Driver.VERBOSE;
import static tools.CNNUtils.*;
import static tools.CommonUtils.checkNotNull;
import static tools.CommonUtils.checkPositive;
import static tools.TensorUtils.tensorSubtract;

/**
 * A convolutional neural network that supports arbitrary convolutional and
 * pooling layers, followed by arbitrarily many fully-connected layers.
 */
public class ConvolutionalNeuralNetwork {
    private final int inputHeight;
    private final int inputWidth;
    private final int maxEpochs;
    private final double initialLearningRate;
    private final List<PlateLayer> plateLayers;
    private final List<FullyConnectedLayer> fullyConnectedLayers;

    private double learningRate;
    /**
     * Passes all images in the dataset through the network and backpropagates the
     * errors.
     */
    private Instance fixedImage = null;

    public ConvolutionalNeuralNetwork(final int inputHeight, final int inputWidth, final List<PlateLayer> plateLayers,
                                      final List<FullyConnectedLayer> fullyConnectedLayers, int maxEpochs, double learningRate) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.plateLayers = plateLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
    }

    /**
     * Returns a new builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Trains the CNN with the given training data and tuning data.
     */
    public void train(final DataSet trainSet, final DataSet tuneSet, final DataSet testSet) {
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            Collections.shuffle(trainSet.getImages());
            trainSingleEpoch(trainSet);

            if (VERBOSE) {
                System.out.printf("Epoch %d completed with train, tune, and test accuracies of %.9f, %.9f, %.9f\n", epoch,
                        test(trainSet, false), test(tuneSet, false), test(testSet, false));
            }

            if (learningRate > .1 * initialLearningRate) {
                learningRate -= (.9 * initialLearningRate) / 1000;
            }
        }
    }

    private void trainSingleEpoch(final DataSet trainSet) {
        if (fixedImage == null) {
            fixedImage = trainSet.getImages().get(0);
        }
        for (Instance img : trainSet.getImages()) {
            int index = trainSet.getImages().indexOf(img);
            if (index == 0) {
                double[] fixedImageOutput = computeOutput(fixedImage);
                for (int i = 0; i < fixedImageOutput.length; i++) {
                    System.out.print(fixedImageOutput[i] + " ");
                }
                System.out.println();
            }

            // First, forward propagate.
            double[] output = computeOutput(img);
            double[] correctOutput = categoryToOneOfN(img.getLabel());

            // Compute initial deltas.
            double[] fcError = tensorSubtract(output, correctOutput, false);
            for (int i = 0; i < fcError.length; i++) {
                fcError[i] *= SIGMOID.applyDerivative(output[i]);
            }

            // Then, propagate error through fully connected layers.
            for (int i = fullyConnectedLayers.size() - 1; i >= 0; i--) {
                fcError = fullyConnectedLayers.get(i).propagateError(fcError, learningRate);
            }

            // Finally, propagate error through plate layers.
            if (plateLayers.size() > 0) {
                ConvolutionLayer lastLayer = (ConvolutionLayer) plateLayers.get(plateLayers.size() - 1);
                List<Plate> plateErrors = packPlates(fcError, lastLayer.outputHeight, lastLayer.outputWidth);
                for (int i = plateLayers.size() - 1; i >= 0; i--) {
                    plateErrors = plateLayers.get(i).propagateError(plateErrors, learningRate);
                }
            }
        }
    }

    /**
     * Returns the prediction accuracy of this classifier on the test set.
     * Here, accuracy is numCorrectlyClassified/numExamples.
     */
    public double test(final DataSet testSet, final boolean verbose) {
        int errCount = 0;
        for (Instance img : testSet.getImages()) {
            Category predicted = classify(img);
            if (!predicted.equals(img.getLabel())) {
                errCount++;
            }

            if (verbose) {
                System.out.printf("Predicted: %s\t\tActual:%s\n", predicted, img.getLabel());
            }
        }

        double accuracy = ((double) (testSet.getSize() - errCount)) / testSet.getSize();
        if (verbose) {
            System.out.printf("Final accuracy was %.9f\n", accuracy);
        }
        return accuracy;
    }

    /**
     * Returns the predicted label for the image.
     */
    public Category classify(final Instance img) {
        double[] outputs = computeOutput(img);
        double maxOutput = -1;
        int maxOutputIndex = -1;
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] > maxOutput) {
                maxOutput = outputs[i];
                maxOutputIndex = i;
            }
        }
        return categories.get(maxOutputIndex);
    }

    /**
     * Propagates the image through the network and returns the last
     * (fully-connected) layer's output.
     */
    private double[] computeOutput(final Instance img) {
        // Pass the input through the plate layers first.
        List<Plate> plates = Arrays.asList(instanceToPlate(img));
        for (PlateLayer layer : plateLayers) {
            plates = layer.computeOutput(plates);
        }

        // Then pass the output through the fully connected layers.
        double[] unpackedPlates = unpackPlates(plates);
        for (FullyConnectedLayer fcLayer : fullyConnectedLayers) {
            unpackedPlates = fcLayer.computeOutput(unpackedPlates);
        }
        return unpackedPlates;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("\n//////\tNETWORK SPECIFICATIONS\t//////\n");
        builder.append(String.format("Input Height: %d\n", inputHeight));
        builder.append(String.format("Input Width: %d\n", inputWidth));
        builder.append(String.format("Number of plate layers: %d\n", plateLayers.size()));
        builder.append(String.format("Number of fully connected hidden layers: %d\n", fullyConnectedLayers.size() - 1));
        builder.append(String.format("Predicts these classes: %s\n", categories));
        builder.append("\n//////\tNETWORK STRUCTURE\t//////\n");
        if (plateLayers.isEmpty()) {
            builder.append("\n------\tNo plate layers!\t------\n");
        } else {
            for (PlateLayer plateLayer : plateLayers) {
                builder.append(plateLayer.toString());
            }
        }
        for (FullyConnectedLayer fcLayer : fullyConnectedLayers) {
            builder.append(fcLayer.toString());
        }
        return builder.toString();
    }

    /**
     * A builder pattern for managing the many parameters of the network.
     */
    public static class Builder {
        private final List<PlateLayer> plateLayers = new ArrayList<>();
        private int inputHeight = 0;
        private int inputWidth = 0;
        private int fullyConnectedWidth = 0;
        private int fullyConnectedDepth = 0;
        private ActivationFunction fcActivation = null;
        private int maxEpochs = 0;
        private double learningRate = 0;

        public Builder setInputHeight(final int height) {
            checkPositive(height, "Input height", false);
            this.inputHeight = height;
            return this;
        }

        public Builder setInputWidth(final int width) {
            checkPositive(width, "Input width", false);
            this.inputWidth = width;
            return this;
        }

        public Builder appendConvolutionLayer(final ConvolutionLayer layer) {
            return appendPlateLayer(layer);
        }

        public Builder appendPoolingLayer(final PoolingLayer layer) {
            return appendPlateLayer(layer);
        }

        private Builder appendPlateLayer(final PlateLayer layer) {
            checkNotNull(layer, "Plate layer");
            this.plateLayers.add(layer);
            return this;
        }

        public Builder setFullyConnectedWidth(final int width) {
            checkPositive(width, "Fully connected width", false);
            this.fullyConnectedWidth = width;
            return this;
        }

        public Builder setFullyConnectedDepth(final int depth) {
            checkPositive(depth, "Fully connected depth", false);
            this.fullyConnectedDepth = depth;
            return this;
        }

        public Builder setFullyConnectedActivationFunction(ActivationFunction fcActivation) {
            checkNotNull(fcActivation, "Fully connected activation function");
            this.fcActivation = fcActivation;
            return this;
        }

        public Builder setMaxEpochs(final int maxEpochs) {
            checkPositive(maxEpochs, "Max epochs", false);
            this.maxEpochs = maxEpochs;
            return this;
        }

        public Builder setLearningRate(final double learningRate) {
            checkPositive(learningRate, "Learning rate", false);
            this.learningRate = learningRate;
            return this;
        }

        public ConvolutionalNeuralNetwork build() {
            checkPositive(inputHeight, "Input height", true);
            checkPositive(inputWidth, "Input width", true);
            checkPositive(fullyConnectedWidth, "Fully connected width", true);
            checkPositive(fullyConnectedDepth, "Fully connected depth", true);
            checkNotNull(fcActivation, "Fully connected activation function");
            checkPositive(maxEpochs, "Max epochs", true);
            checkPositive(learningRate, "Learning rate", true);

            // Given input dimensions, determine how many plates will be output by
            // the last plate layer, and the dimensions of those plates.
            // Note that if there are no plate layers, then this result defaults to
            // imageHeight * imageWidth, which is what we need in that case.
            int outputHeight = inputHeight;
            int outputWidth = inputWidth;
            int numOutputs = 3;
            for (PlateLayer plateLayer : plateLayers) {
                outputHeight = plateLayer.calculateOutputHeight(outputHeight);
                outputWidth = plateLayer.calculateOutputWidth(outputWidth);
                numOutputs = plateLayer.calculateNumPlateOutputs(numOutputs);
            }

            List<FullyConnectedLayer> fullyConnectedLayers = new ArrayList<>(fullyConnectedDepth);

            // Always have at least one hidden layer - add it first.
            // TODO: Make the fully-connected activation function a parameter.
            int numInputs = outputWidth * outputHeight * numOutputs;
            numInputs = (plateLayers.size() > 0)
                    ? numInputs * ((ConvolutionLayer) plateLayers.get(plateLayers.size() - 1)).getConvolutions().size()
                    : numInputs;
            fullyConnectedLayers.add(FullyConnectedLayer.newBuilder().setActivationFunction(fcActivation).setNumInputs(numInputs).setNumNodes(fullyConnectedWidth).build());

            // Add the other hidden layers.
            for (int i = 0; i < fullyConnectedDepth - 1; i++) {
                fullyConnectedLayers.add(FullyConnectedLayer.newBuilder().setActivationFunction(fcActivation).setNumInputs(fullyConnectedWidth).setNumNodes(fullyConnectedWidth).build());
            }

            // Add the output layer.
            fullyConnectedLayers.add(FullyConnectedLayer.newBuilder().setActivationFunction(fcActivation).setNumInputs(fullyConnectedWidth).setNumNodes(Category.values().length).build());

            return new ConvolutionalNeuralNetwork(inputHeight, inputWidth, plateLayers, fullyConnectedLayers, maxEpochs, learningRate);
        }
    }
}
