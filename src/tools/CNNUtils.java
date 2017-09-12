package tools;

import cnn.Plate;
import data.Instance;

import java.util.ArrayList;
import java.util.List;

import static tools.CommonUtils.checkNotEmpty;

public class CNNUtils {

    /**
     * Converts an instance to an array of plates.
     */
    public static Plate[] instanceToPlate(final Instance instance) {
        return new Plate[]{new Plate(intImgToDoubleImg(instance.getRedChannel())),
                new Plate(intImgToDoubleImg(instance.getGreenChannel())),
                new Plate(intImgToDoubleImg(instance.getBlueChannel()))
        };
    }

    public static double[][] intImgToDoubleImg(final int[][] intImg) {
        double[][] dblImg = new double[intImg.length][intImg[0].length];
        for (int i = 0; i < dblImg.length; i++) {
            for (int j = 0; j < dblImg[i].length; j++) {
                dblImg[i][j] = ((double) 255 - intImg[i][j]) / 255;
            }
        }
        return dblImg;
    }

    /**
     * Unpack the plates into a single, 1D double array. Used to connect the plate
     * layers with the fully connected layers.
     */
    public static double[] unpackPlates(final List<Plate> plates) {
        checkNotEmpty(plates, "Plates to pack", false);
        int flattenedPlateSize = plates.get(0).getTotalNumValues();
        double[] result = new double[flattenedPlateSize * plates.size()];
        for (int i = 0; i < plates.size(); i++) {
            System.arraycopy(plates.get(i).as1DArray(), 0 /* Copy the whole flattened plate! */, result,
                    i * flattenedPlateSize, flattenedPlateSize);
        }
        return result;
    }

    /**
     * Pack the 1D double array into a list of plates (3D double tensors).
     */
    public static List<Plate> packPlates(final double[] unpackedPlates, final int plateHeight, final int plateWidth) {
        List<Plate> plates = new ArrayList<>();
        int k = 0;
        while (k < unpackedPlates.length) {
            double[][] unpackedPlate = new double[plateHeight][plateWidth];
            for (int i = 0; i < plateHeight; i++) {
                for (int j = 0; j < plateWidth; j++) {
                    unpackedPlate[i][j] = unpackedPlates[k++];
                }
            }
            plates.add(new Plate(unpackedPlate));
        }
        return plates;
    }
}
