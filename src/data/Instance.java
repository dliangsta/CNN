package data;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Instance {

    private final BufferedImage image;
    private final Category label;
    private final int width, height;

    private int[][] red_channel, green_channel, blue_channel;

    public Instance(final BufferedImage image, final String label) {
        this.image = image;
        this.label = Category.valueOf(label);
        width = image.getWidth();
        height = image.getHeight();

        // Get separate rgb channels.
        red_channel = new int[height][width];
        green_channel = new int[height][width];
        blue_channel = new int[height][width];

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                Color c = new Color(this.image.getRGB(col, row));
                red_channel[row][col] = c.getRed();
                green_channel[row][col] = c.getGreen();
                blue_channel[row][col] = c.getBlue();
            }
        }
    }

    public int[][] getRedChannel() {
        return red_channel;
    }

    public int[][] getGreenChannel() {
        return green_channel;
    }

    public int[][] getBlueChannel() {
        return blue_channel;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Category getLabel() {
        return label;
    }
}
