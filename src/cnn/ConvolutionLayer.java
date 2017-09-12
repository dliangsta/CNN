package cnn;

import java.util.ArrayList;
import java.util.List;

import static cnn.ActivationFunction.RELU;
import static tools.CommonUtils.*;
import static tools.TensorUtils.tensorAdd;

/**
 * A layer that performs n convolutions. Uses ReLU for activation.
 */
public class ConvolutionLayer extends PlateLayer {

    private final List<List<Plate>> convolutions;
    private List<Plate> previousInput;
    private List<Plate> previousOutput;

    private ConvolutionLayer(final List<List<Plate>> convolutions) {
        this.convolutions = convolutions;
    }

    /**
     * Returns a new builder.
     */
    public static Builder newBuilder() {
        return new Builder();
    }

    @Override
    public List<Plate> computeOutput(final List<Plate> input) {
        checkNotNull(input, "Convolution layer input");
        checkNotEmpty(input, "Convolution layer input", false);
        previousInput = input;
        // Convolve each input with each mask.
        List<Plate> outputPlates = new ArrayList<>();
        for (int i = 0; i < convolutions.size(); i++) {
            double[][] output = new double[input.get(0).getHeight() - getConvolutionHeight()
                    + 1][input.get(0).getWidth() - getConvolutionWidth() + 1];
            // convolve each input image, sum the output, add the new plate
            for (int j = 0; j < getConvolutionDepth(); j++) {
                Plate convolution = convolutions.get(i).get(j);
                tensorAdd(output, input.get(j).convolve(convolution).getValues(), true);
                input.get(j);
            }
            outputPlates.add((new Plate(output).applyActivation(RELU)));
        }
        previousOutput = outputPlates;
        return outputPlates;
    }

    @Override
    public List<Plate> propagateError(final List<Plate> errors, final double learningRate) {
        if (errors.size() != previousOutput.size() || previousInput.isEmpty()) {
            if (errors.size() != previousOutput.size()) {
                System.err.println(String.format("Error size (%d) and previous output size (%d) mismatch!",
                        errors.size(), previousOutput.size()));
            } else {
                System.err.println("Previous input is empty!");
            }
            System.err.println(this.toString());
            throw new IllegalArgumentException("Bad propagation state.");
        }
        // Stores the delta values for all the plates in current layer
        List<Plate> deltaOutput = new ArrayList<>(errors.size());

        // Total error
        double[][][] error = new double[errors.size()][][];
        double[][][] delta = new double[error.length][][];

        for (int i = 0; i < errors.size(); i++) {
            error[i] = new double[previousInput.get(0).getHeight()][previousInput.get(0).getWidth()];
            delta[i] = new double[error[i].length][error[i][0].length];
            for (int kernel = 0; kernel < getConvolutionDepth(); kernel++) {
                for (int row = 0; row < error[i].length - getConvolutionHeight(); row++) {
                    for (int col = 0; col < error[i][row].length - getConvolutionWidth(); col++) {
                        for (int kernelRow = 0; kernelRow < getConvolutionHeight(); kernelRow++) {
                            for (int kernelCol = 0; kernelCol < getConvolutionWidth(); kernelCol++) {
                                error[i][row + kernelRow][col + kernelCol] += errors.get(i).valueAt(row, col) * convolutions.get(i).get(kernel).valueAt(kernelRow, kernelCol);
                            }
                        }
                    }
                }
                for (int row = 0; row < error[i].length; row++) {
                    for (int col = 0; col < error[i][row].length; col++) {
                        delta[i][row][col] += error[i][row][col] * RELU.applyDerivative(previousInput.get(kernel).valueAt(row, col));
                    }
                }
            }
            deltaOutput.add(new Plate(delta[i]));
        }
        return deltaOutput;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("\n------\tConvolution Layer\t------\n\n");
        builder.append(String.format("Convolution Size: %dx%dx%d\n", getConvolutionDepth(), getConvolutionHeight(),
                getConvolutionWidth()));
        builder.append(String.format("Number of convolutions: %d\n", convolutions.size()));
        builder.append(String.format("Input size: %d x %d x %d\n", inputDepth, inputHeight, inputWidth));
        builder.append(String.format("Output size: %d x %d x %d\n", outputDepth, outputHeight, outputWidth));
        builder.append("Activation Function: RELU\n");
        builder.append("\n\t------------\t\n");
        return builder.toString();
    }

    /**
     * Returns the number of different convolutions in this layer.
     */
    public int numConvolutions() {
        return convolutions.size();
    }

    /**
     * Returns the convolutions for this layer.
     */
    public List<List<Plate>> getConvolutions() {
        return convolutions;
    }

    /**
     * Returns the number of channels in each convolution.
     */
    public int getConvolutionDepth() {
        return convolutions.get(0).size();
    }

    /**
     * Returns the convolution height.
     */
    public int getConvolutionHeight() {
        return convolutions.get(0).get(0).getHeight();
    }

    /**
     * Returns the convolution width.
     */
    public int getConvolutionWidth() {
        return convolutions.get(0).get(0).getWidth();
    }

    @Override
    public int calculateNumPlateOutputs(final int numInputs) {
        this.numInputs = numInputs;
        numOutputs = numInputs / getConvolutionDepth();
        return numOutputs;
    }

    @Override
    public int calculateOutputHeight(final int inputHeight) {
        this.inputHeight = inputHeight;
        outputHeight = inputHeight - getConvolutionHeight() + 1;
        return outputHeight;
    }

    @Override
    public int calculateOutputWidth(final int inputWidth) {
        this.inputWidth = inputWidth;
        outputWidth = inputWidth - getConvolutionWidth() + 1;
        return outputWidth;
    }

    /**
     * A simple builder pattern for managing the layer's parameters at construction.
     */
    public static class Builder {
        private int numChannels = 0;
        private int convolutionHeight = 0;
        private int convolutionWidth = 0;
        private int numConvolutions = 0;

        private Builder() {
        }

        private static double[][] createRandomConvolution(final int height, final int width) {
            double[][] plateValues = new double[height][width];
            for (int i = 0; i < plateValues.length; i++) {
                for (int j = 0; j < plateValues[i].length; j++) {
                    plateValues[i][j] = rng.nextGaussian();
                }
            }
            return plateValues;
        }

        public Builder setConvolutionSize(final int numChannels, final int height, final int width) {
            checkPositive(numChannels, "Convolution channels", false);
            checkPositive(height, "Convolution height", false);
            checkPositive(width, "Convolution width", false);
            this.numChannels = numChannels;
            this.convolutionHeight = height;
            this.convolutionWidth = width;
            return this;
        }

        public Builder setNumConvolutions(final int numConvolutions) {
            checkPositive(numConvolutions, "Number of convolutions", false);
            this.numConvolutions = numConvolutions;
            return this;
        }

        public ConvolutionLayer build() {
            checkPositive(numChannels, "Convolution channels", true);
            checkPositive(convolutionHeight, "Convolution height", true);
            checkPositive(convolutionWidth, "Convolution width", true);
            checkPositive(numConvolutions, "Number of convolutions", true);

            List<List<Plate>> convolutions = new ArrayList<>();
            for (int i = 0; i < numConvolutions; i++) {
                List<Plate> channelConvolutions = new ArrayList<>();
                for (int j = 0; j < numChannels; j++) {
                    channelConvolutions.add(new Plate(createRandomConvolution(convolutionHeight, convolutionWidth)));
                }
                convolutions.add(channelConvolutions);
            }
            return new ConvolutionLayer(convolutions);
        }
    }
}
