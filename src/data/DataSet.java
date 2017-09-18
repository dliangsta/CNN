package data;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static data.Category.categories;
import static tools.CommonUtils.rng;

/**
 * This is the dataset class that loads in the whole dataset.
 */
public class DataSet {
    private static final boolean createExtraTrainingExamples = true;
    private static int MAX_INSTANCES = 50;
    private static boolean FAST = false;
    private static int imageSize;

    private final List<Instance> instances;
    private final List<Instance> trainsetExtras;

    protected static final double shiftProbNumerator = 6.0; // 6.0 is the 'default.'
    protected static final double probOfKeepingShiftedTrainsetImage = (shiftProbNumerator / 48.0); // This 48 is also embedded elsewhere!
    protected static final boolean perturbPerturbedImages = false;

    public DataSet(final String directoryName) {
        final File dir = new File(directoryName);
        instances = new ArrayList<>();
        trainsetExtras = new ArrayList<>();
        for (File file : dir.listFiles()) {
            // check all files
            if (!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            // String path = file.getAbsolutePath();
            BufferedImage img, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format: label_image_XXXX(4 digits) though this
                // code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");

                // Resize the image if requested. Any resizing allowed, but should really be one
                // of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }

                Instance instance = new Instance(scaledBI == null ? img : scaledBI, directoryName, name.substring(0, locationOfUnderscoreImage).toUpperCase());

                instances.add(instance);
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }
        }

        if (FAST) {
            while (instances.size() > MAX_INSTANCES) {
                instances.remove((int) (rng.nextDouble() * instances.size()));
            }
        }

        if (createExtraTrainingExamples && directoryName.equals("images/trainset/")) {
            // Flipping watches will mess up the digits on the watch faces, but that probably is ok.
            for (Instance origTrainImage : instances) {
                createMoreImagesFromThisImage(origTrainImage, 1.00);
            }

            int[] countOfCreatedTrainingImages = new int[Category.values().length];
            int count_trainsetExtrasKept = 0;
            for (Instance createdTrainImage : trainsetExtras) {
                // Keep more of the less common categories?
                double probOfKeeping = 1.0;

                // Trainset counts: airplanes=127, butterfly=55, flower=114, piano=61, starfish=51, watch=146
                if ("airplanes".equals(createdTrainImage.getLabel()))
                    probOfKeeping = 0.66; // No flips, so fewer created.
                else if ("BUTTERFLY".equals(createdTrainImage.getLabel()))
                    probOfKeeping = 1.00; // No top-bottom flips, so fewer created.
                else if ("FLOWER".equals(createdTrainImage.getLabel()))
                    probOfKeeping = 0.66; // No top-bottom flips, so fewer created.
                else if ("GRAND_PIANO".equals(createdTrainImage.getLabel()))
                    probOfKeeping = 1.00; // No flips, so fewer created.
                else if ("STARFISH".equals(createdTrainImage.getLabel()))
                    probOfKeeping = 1.00; // No top-bottom flips, so fewer created.
                else if ("WATCH".equals(createdTrainImage.getLabel()))
                    probOfKeeping = 0.50; // Already have a lot of these.


                if (rng.nextDouble() <= probOfKeeping) {
                    countOfCreatedTrainingImages[categories.indexOf(createdTrainImage.getLabel())]++;
                    count_trainsetExtrasKept++;
                    instances.add(createdTrainImage);
                }
            }

            for (Category cat : Category.values()) {
                System.out.println(" Kept " + countOfCreatedTrainingImages[cat.ordinal()] + " 'tweaked' images of " + cat + ".");
            }

            System.out.println("Created a total of " + trainsetExtras.size() + " new training examples and kept " + count_trainsetExtrasKept + ".");
            System.out.println("The trainset NOW contains " + this.instances.size() + " examples. ");
        }
    }

    private void createMoreImagesFromThisImage(Instance trainImage, double probOfKeeping) {
        if (!"airplanes".equals(trainImage.getLabel()) &&  // Airplanes all 'face' right and up, so don't flip left-to-right or top-to-bottom.
                !"grand_piano".equals(trainImage.getLabel())) {  // Ditto for pianos.

            if (trainImage.getProvenance() != Instance.HowCreated.FlippedLeftToRight && rng.nextDouble() <= probOfKeeping)
                trainsetExtras.add(trainImage.flipImageLeftToRight());

            if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't flip to-to-bottom.
                    !"flower".equals(trainImage.getLabel()) &&  // Ditto for flowers.
                    !"starfish".equals(trainImage.getLabel())) {  // Star fish are standardized to 'point up.
                if (trainImage.getProvenance() != Instance.HowCreated.FlippedTopToBottom && rng.nextDouble() <= probOfKeeping)
                    trainsetExtras.add(trainImage.flipImageTopToBottom());
            }
        }
        boolean rotateImages = true;
        if (rotateImages && trainImage.getProvenance() != Instance.HowCreated.Rotated) {
            //    Instance rotated = origTrainImage.rotateImageThisManyDegrees(3);
            //    origTrainImage.display2D(origTrainImage.getGrayImage());
            //    rotated.display2D(              rotated.getGrayImage()); waitForEnter();

            if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(3));
            if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-3));
            if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(4));
            if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-4));
            if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't rotate too much.
                    !"flower".equals(trainImage.getLabel()) &&  // Ditto for flowers and starfish.
                    !"starfish".equals(trainImage.getLabel())) {
                if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(5));
                if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-5));
            } else {
                if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(2));
                if (rng.nextDouble() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-2));
            }
        }
        // Would be good to also shift and rotate the flipped examples, but more complex code needed.
        if (trainImage.getProvenance() != Instance.HowCreated.Shifted) {
            for (int shiftX = -3; shiftX <= 3; shiftX++) {
                for (int shiftY = -3; shiftY <= 3; shiftY++) {
                    // Only keep some of these, so these don't overwhelm the flipped and rotated examples when down sampling below.
                    if ((shiftX != 0 || shiftY != 0) && rng.nextDouble() <= probOfKeepingShiftedTrainsetImage * probOfKeeping)
                        trainsetExtras.add(trainImage.shiftImage(shiftX, shiftY));
                }
            }
        }

//        for (Instance instance : trainsetExtras) {
//            instances.add(instance);
//        }
    }


    public static void setImageSize(final int imageSize) {
        DataSet.imageSize = imageSize;
    }

    // get the size of the dataset
    public int getSize() {
        return instances.size();
    }

    // Return the list of images.
    public List<Instance> getImages() {
        return instances;
    }
}
