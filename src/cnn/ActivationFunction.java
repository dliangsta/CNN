package cnn;

import java.util.function.Function;

/**
 * Represents activation functions for any node.
 */
public enum ActivationFunction {
    RELU(/* function */ x -> Math.max(x, 0.000001 * x),
            /* derivative */ x -> (x > 0) ? 1.0 : 0.000001),
    SIGMOID(/* function */ x -> 1 / (1 + Math.pow(Math.E, -x)),
            /* derivative */ x -> x * (1 - x));

    private final Function<Double, Double> func;
    private final Function<Double, Double> derivative;

    ActivationFunction(final Function<Double, Double> func, final Function<Double, Double> derivative) {
        this.func = func;
        this.derivative = derivative;
    }

    /**
     * Applies the activation function.
     */
    public double apply(final double x) {
        return func.apply(x);
    }

    /**
     * Evaluates the derivative at x.
     * <p>
     * NOTE: Assumes that x is a value that has already been passed through the
     * activation function. (These derivatives all depend on the value at the activation function.)
     */
    public double applyDerivative(final double x) {
        return derivative.apply(x);
    }
}
