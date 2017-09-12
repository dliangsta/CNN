package data;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static tools.CommonUtils.rng;

/**
 * This is the dataset class that loads in the whole dataset.
 */
public class DataSet {

    private static int MAX_INSTANCES = 50;
    private static boolean FAST = !true;
    private static int imageSize;

    private final ArrayList<Instance> instances;


    public DataSet(final String directoryName) {
        final File dir = new File(directoryName);
        instances = new ArrayList<>();
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

                Instance instance = new Instance(scaledBI == null ? img : scaledBI,
                        name.substring(0, locationOfUnderscoreImage).toUpperCase());

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
