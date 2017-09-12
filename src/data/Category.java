package data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public enum Category {
    AIRPLANES, BUTTERFLY, FLOWER, GRAND_PIANO, STARFISH, WATCH;
    // Store the categories as strings.
    public static List<Category> categories = new ArrayList<>(Arrays.asList(values()));

    /**
     * Returns the one of n index of a category label.
     */
    public static double[] categoryToOneOfN(final Category category) {
        double[] correctOutput = new double[categories.size()];
        correctOutput[categories.indexOf(category)] = 1;
        return correctOutput;
    }
};