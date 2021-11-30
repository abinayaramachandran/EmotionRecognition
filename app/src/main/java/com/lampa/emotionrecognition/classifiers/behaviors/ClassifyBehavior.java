package com.lampa.emotionrecognition.classifiers.behaviors;

import java.nio.ByteBuffer;

public interface ClassifyBehavior {
    float[][] classify(ByteBuffer input);
}
