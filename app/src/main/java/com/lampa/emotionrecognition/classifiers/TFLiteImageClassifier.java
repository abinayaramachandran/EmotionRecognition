package com.lampa.emotionrecognition.classifiers;

import com.lampa.emotionrecognition.classifiers.behaviors.TFLiteImageClassification;
import com.lampa.emotionrecognition.utils.ImageUtils;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.util.Formatter;
import java.util.HashMap;
import java.util.Map;

// Image classifier that uses tflite format
public class TFLiteImageClassifier extends TFLiteClassifier {

    public TFLiteImageClassifier(Activity activity, AssetManager assetManager, String modelFileName, String[] labels) {
        super(activity, assetManager, modelFileName, labels);

        classifyBehavior = new TFLiteImageClassification(mInterpreter);
    }

    public Map<String, Float> classify(Bitmap imageBitmap, boolean useFilter) {
        float[] preprocessedImage = preprocessImage(imageBitmap, useFilter);

        return classify(preprocessedImage);
    }

    private Map<String, Float> classify(float[] input) {
        float[][] outputArr = classifyBehavior.classify(input);

        // Checked compliance with the array of strings specified in the constructor
        if (mLabels.size() != outputArr[0].length) {
            Formatter formatter = new Formatter();

            throw new IllegalArgumentException(formatter.format(
                    "labels array length must be equal to %1$d, but actual length is %2$d",
                    outputArr[0].length,
                    mLabels.size()
            ).toString());
        }

        Map<String, Float> outputMap = new HashMap<>();

        String predictedLabel;
        float probability;
        for (int i = 0; i < outputArr[0].length; i++) {
            predictedLabel = mLabels.get(i);
            probability = outputArr[0][i];

            outputMap.put(predictedLabel, probability);
        }

        return outputMap;
    }

    private float[] preprocessImage(Bitmap imageBitmap, boolean useFilter) {
        //Scale an image

//        ImageProcessor imageProcessor =
//                new ImageProcessor.Builder()
//                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
//                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
//                        .add(new Rot90Op(numRoration))
//                        .add(getPreprocessNormalizeOp())
//                        .build();
//        return imageProcessor.process(inputImageBuffer);


        Bitmap scaledImage = Bitmap.createScaledBitmap(
                imageBitmap,
                InterpreterImageParams.getInputImageWidth(mInterpreter),
                InterpreterImageParams.getInputImageHeight(mInterpreter),
                useFilter);

//        Bitmap scaledImage = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888);


        // Translate an image to greyscale format
        int[] greyScaleImage = ImageUtils.toGreyScale(scaledImage);
        Log.d("Test " , String.valueOf(greyScaleImage.length));
        // Translate an image to normalized float format [0f, 1f]
        float[] preprocessedImage = new float[greyScaleImage.length];
        for (int i = 0; i < preprocessedImage.length; i++) {
            preprocessedImage[i] = greyScaleImage[i] / 255.0f;
        }

        return preprocessedImage;
    }
}
