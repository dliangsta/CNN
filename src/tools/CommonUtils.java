package tools;

import java.util.Collection;
import java.util.Random;

public class CommonUtils {

    public static Random rng = new Random(638 * 838);

    public static double random() {
        return rng.nextDouble();
    }

    public static int randomInInterval(final int lower, final int upper) {
        return lower + (int) Math.floor(random() * (upper - lower));
    }

    public static int random0toNminus1(final int upper) {
        return randomInInterval(0, upper);
    }

    public static double getRandomWeight(final int fanin, final int fanout) {
        double range = Math.max(Double.MIN_VALUE, 4.0 / Math.sqrt(6.0 * (fanin + fanout)));
        return (2.0 * random() - 1.0) * range;
    }

    public static void checkVectorNotNullOrEmpty(final double[] vector) {
        checkNotNull(vector, "Vector arg");
        checkPositive(vector.length, "Vector length", false);
    }

    public static void checkVectorDimensionsMatch(final double[] v1, final double[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException(
                    String.format("Lengths %d and %d do not match for inner product.\n", v1.length, v2.length));
        }
    }

    /**
     * Verifies that the tensor is not null and that all 3 dimensions have length >
     * 0.
     */
    public static void checkTensorNotNullOrEmpty(final double[][][] tensor) {
        checkNotNull(tensor, "Tensor arg");
        checkPositive(tensor.length, "Tensor dimension 1", false);
        checkPositive(tensor[0].length, "Tensor dimension 2", false);
        checkPositive(tensor[0][0].length, "Tensor dimension 3", false);
    }

    /**
     * Verifies that the tensors have the same dimensions.
     */
    public static void checkTensorDimensionsMatch(final double[][][] t1, final double[][][] t2) {
        if (t1.length != t2.length || t1[0].length != t2[0].length || t1[0][0].length != t2[0][0].length) {
            throw new IllegalArgumentException(
                    String.format("Tensor dimensions do not match...\tT1:%dx%dx%d\tT2:%dx%dx%d\n", t1.length,
                            t1[0].length, t1[0][0].length, t2.length, t2[0].length, t2[0][0].length));
        }
    }

    /**
     * Verifies that the value with the given name is in the range [min, max).
     */
    public static void checkValueInRange(final int val, final int min, final int max, final String name) {
        if (val < min || val >= max) {
            throw new IllegalArgumentException(
                    String.format("%s was %d, but should be in range [%d, %d)", name, val, min, max));
        }
    }

    /**
     * Verifies that the object with the given name is not null.
     */
    public static void checkNotNull(Object obj, String name) {
        if (obj == null) {
            throw new NullPointerException(String.format("%s was null!", name));
        }
    }

    /**
     * Verifies that the value with the given name is not null.
     * <p>
     * The boolean parameter specifies which type of exception to throw.
     */
    public static void checkPositive(double val, String name, boolean sourceIsStateful) {
        if (val <= 0) {
            if (sourceIsStateful) {
                throw new IllegalStateException(String.format("%s was not set!", name));
            } else {
                throw new IllegalArgumentException(String.format("%s must be positive!", name));
            }
        }
    }

    /**
     * Checks that the collection with the given name is nonempty.
     * <p>
     * Again, the boolean parameter specifies the type of exception to throw.
     */
    public static void checkNotEmpty(Collection<?> coll, String name, boolean sourceIsStateful) {
        if (coll.isEmpty()) {
            if (sourceIsStateful) {
                throw new IllegalStateException(String.format("%s must have at least one value!", name));
            } else {
                throw new IllegalArgumentException(String.format("%s must be nonempty!", name));
            }
        }
    }
}
