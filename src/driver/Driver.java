package driver;

import cnn.ConvolutionLayer;
import cnn.ConvolutionalNeuralNetwork;
import cnn.PoolingLayer;
import data.DataSet;

import static cnn.ActivationFunction.SIGMOID;

public final class Driver {

    public static final boolean VERBOSE = true;

    private static int imageSize = 32;

//    private static double etaabs = 0.00001;
    private static double eta = .01;

    public static void main(final String[] args) {
        String trainDirectory = "images/trainset/";
        String tuneDirectory = "images/tuneset/";
        String testDirectory = "images/testset/";

        // parse command line args
        if (args.length > 5) {
            System.err.println(
                    "Usage error: java Main <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
            System.exit(1);
        }
        if (args.length >= 1) {
            trainDirectory = args[0];
        }
        if (args.length >= 2) {
            tuneDirectory = args[1];
        }
        if (args.length >= 3) {
            testDirectory = args[2];
        }
        if (args.length >= 4) {
            imageSize = Integer.parseInt(args[3]);
        }

        // load data
        DataSet.setImageSize(imageSize);
        DataSet trainSet = new DataSet(trainDirectory);
        DataSet tuneSet = new DataSet(tuneDirectory);
        DataSet testSet = new DataSet(testDirectory);

        // build CNN
        ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork.builder().setInputHeight(imageSize)
                .setInputWidth(imageSize)
                .appendConvolutionLayer(ConvolutionLayer.newBuilder().setConvolutionSize(3, 5, 5).setNumConvolutions(20).build())
                .appendPoolingLayer(PoolingLayer.newBuilder().setWindowSize(2, 2).build())
                .appendConvolutionLayer(ConvolutionLayer.newBuilder().setConvolutionSize(1, 5, 5).setNumConvolutions(20).build())
                .appendPoolingLayer(PoolingLayer.newBuilder().setWindowSize(2, 2).build())
                .appendConvolutionLayer(ConvolutionLayer.newBuilder().setConvolutionSize(1, 3, 3).setNumConvolutions(20).build())
                .setFullyConnectedDepth(1).setFullyConnectedWidth(300).setFullyConnectedActivationFunction(SIGMOID)
                .setLearningRate(eta).setMaxEpochs(10000).build();

        System.out.println("******\tDeep CNN constructed." + " The structure is described below.\t******");
        System.out.println(cnn);

        // train CNN
        System.out.println(
                "******\tDeep CNN training has begun." + " Updates will be provided after each epoch.\t******");
        cnn.train(trainSet, tuneSet, testSet);

        System.out.println("\n******\tDeep CNN testing has begun.\t******");
        System.out.println(cnn.test(testSet, true) + "% accuracy");
        return;
    }
}