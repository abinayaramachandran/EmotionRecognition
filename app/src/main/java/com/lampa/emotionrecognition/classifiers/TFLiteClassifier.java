package com.lampa.emotionrecognition.classifiers;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.FileUtils;
import org.tensorflow.lite.support.common.FileUtil;

//import com.google.firebase.firestore.util.FileUtil;
import com.lampa.emotionrecognition.classifiers.behaviors.ClassifyBehavior;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

// Abstract classifier using tflite format
public abstract class TFLiteClassifier {
    protected AssetManager mAssetManager;

    protected Interpreter mInterpreter;

    protected Interpreter.Options mTFLiteInterpreterOptions;

    protected List<String> mLabels;

    protected ClassifyBehavior classifyBehavior;

    public TFLiteClassifier(Activity activity, AssetManager assetManager, String modelFileName, String[] labels) {
        mAssetManager = assetManager;

        GpuDelegate delegate = new GpuDelegate();
        mTFLiteInterpreterOptions = new Interpreter.Options().addDelegate(delegate);
        try {
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, modelFileName);
            mInterpreter = new Interpreter(tfliteModel);
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        mLabels = new ArrayList<>(Arrays.asList(labels));
    }

    public MappedByteBuffer loadModel(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = mAssetManager.openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());

        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Close the interpreter to avoid memory leaks
    public void close() {
        mInterpreter.close();
    }

    public Interpreter getInterpreter() {
        return mInterpreter;
    }

    public List<String> getLabels() {
        return mLabels;
    }

    public void setLabels(List<String> labels) {
        mLabels = labels;
    }
}
