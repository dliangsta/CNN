package cnn;

import java.util.List;

/**
 * Interface for passing plates between conv and pool layers.
 */
public abstract class PlateLayer {

    protected int numInputs;
    protected int inputHeight;
    protected int inputWidth;
    protected int inputDepth = 3;
    protected int numOutputs;
    protected int outputHeight;
    protected int outputWidth;
    protected int outputDepth = 3;

    /**
     * Given the number of inputs, return how many plates this layer will output.
     */
    public abstract int calculateNumPlateOutputs(final int numInputs);

    /**
     * Given the height of the input, return the height of the output.
     */
    public abstract int calculateOutputHeight(final int inputHeight);

    /**
     * Given the width of the input, return the width of the output.
     */
    public abstract int calculateOutputWidth(final int inputWidth);

    /**
     * Pass the given plates through this layer.
     */
    public abstract List<Plate> computeOutput(final List<Plate> input);

    /**
     * Propagate errors (deltas stored in plates) through this layer, and return the
     * deltas for the next layer.
     */
    public abstract List<Plate> propagateError(final List<Plate> errors, final double learningRate);
}
