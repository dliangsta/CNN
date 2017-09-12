package tools;

import static tools.CommonUtils.*;

/**
 * Utility methods and objects used throughout the network.
 */
public final class TensorUtils {

    /**
     * Performs v1 * v2^T.
     */
    public static double[][] outerProduct(final double[] v1, final double[] v2) {
        checkVectorNotNullOrEmpty(v1);
        checkVectorNotNullOrEmpty(v2);
        double[][] result = new double[v1.length][v2.length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = v1[i] * v2[j];
            }
        }
        return result;
    }

    /**
     * Computes the inner product of two vectors.
     */
    public static double innerProduct(final double[] v1, final double[] v2) {
        checkVectorNotNullOrEmpty(v1);
        checkVectorNotNullOrEmpty(v2);
        checkVectorDimensionsMatch(v1, v2);
        double result = 0;
        for (int i = 0; i < v1.length; i++) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    /**
     * Performs vector scalar multiplication. See description for 3D version.
     */
    public static double[] scalarMultiply(final double scalar, final double[] vector, final boolean inline) {
        return scalarMultiply(scalar, new double[][][]{{vector}}, inline)[0][0];
    }

    /**
     * Performs tensor scalar multiplication. See description for 3D version.
     */
    public static double[][] scalarMultiply(final double scalar, final double[][] tensor, final boolean inline) {
        return scalarMultiply(scalar, new double[][][]{tensor}, inline)[0];
    }

    /**
     * Performs tensor scalar multiplication.
     * If inline is true, this method directly mutates the given tensor.
     */
    public static double[][][] scalarMultiply(final double scalar, final double[][][] tensor, final boolean inline) {
        checkTensorNotNullOrEmpty(tensor);
        double[][][] result = inline ? tensor : new double[tensor.length][tensor[0].length][tensor[0][0].length];
        for (int i = 0; i < tensor.length; i++) {
            for (int j = 0; j < tensor[i].length; j++) {
                for (int k = 0; k < tensor[i][j].length; k++) {
                    result[i][j][k] = tensor[i][j][k] * scalar;
                }
            }
        }
        return result;
    }

    /**
     * Performs v1 - v2 (for vectors). See description for 3D version.
     */
    public static double[] tensorSubtract(final double[] v1, final double[] v2, final boolean inline) {
        return tensorSubtract(new double[][][]{{v1}}, new double[][][]{{v2}}, inline)[0][0];
    }

    /**
     * Performs m1 - m2 (for matrices). See description for 3D version.
     */
    public static double[][] tensorSubtract(final double[][] m1, final double[][] m2, final boolean inline) {
        return tensorSubtract(new double[][][]{m1}, new double[][][]{m2}, inline)[0];
    }

    /**
     * Performs t1 - t2 (for 3D tensors). See description for 3D tensorAdd.
     */
    public static double[][][] tensorSubtract(final double[][][] t1, final double[][][] t2, final boolean inline) {
        // return tensorAdd(t1, scalarMultiply(-1, t2, inline), inline);
        checkTensorNotNullOrEmpty(t1);
        checkTensorNotNullOrEmpty(t2);
        checkTensorDimensionsMatch(t1, t2);

        double[][][] result = inline ? t1 : new double[t1.length][t1[0].length][t1[0][0].length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                for (int k = 0; k < result[i][j].length; k++) {
                    result[i][j][k] = t1[i][j][k] - t2[i][j][k];
                }
            }
        }
        return result;
    }

    /**
     * Performs v1 + v2 (for vectors). See description for 3D version.
     */
    public static double[] tensorAdd(final double[] v1, final double[] v2, final boolean inline) {
        return tensorAdd(new double[][][]{{v1}}, new double[][][]{{v2}}, inline)[0][0];
    }

    /**
     * Performs m1 + m2 (for matrices). See description for 3D version.
     */
    public static double[][] tensorAdd(final double[][] m1, final double[][] m2, final boolean inline) {
        return tensorAdd(new double[][][]{m1}, new double[][][]{m2}, inline)[0];
    }

    /**
     * Performs t1 + t2 (for 3D tensors).
     * <p>
     * If inline is true, this method directly mutates t1.
     */
    public static double[][][] tensorAdd(final double[][][] t1, final double[][][] t2, final boolean inline) {
        checkTensorNotNullOrEmpty(t1);
        checkTensorNotNullOrEmpty(t2);
        checkTensorDimensionsMatch(t1, t2);
        double[][][] result = inline ? t1 : new double[t1.length][t1[0].length][t1[0][0].length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                for (int k = 0; k < result[i][j].length; k++) {
                    result[i][j][k] = t1[i][j][k] + t2[i][j][k];
                }
            }
        }
        return result;
    }
}
