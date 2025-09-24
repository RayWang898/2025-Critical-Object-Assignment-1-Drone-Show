//!!rrr
package com.dji.sdk.sample.internal.controller;

import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.media.MediaFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;

import androidx.appcompat.app.AppCompatActivity;

import com.dji.sdk.sample.R;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import dji.sdk.codec.DJICodecManager;
import dji.sdk.camera.VideoFeeder;

public class HumanDetection extends AppCompatActivity implements TextureView.SurfaceTextureListener {

    private TextureView textureView;
    private DJICodecManager codecManager;
    private VideoFeeder.VideoDataListener videoDataListener;
    private OverlayView overlayView;

    private Net net;
    private static final String[] classNames = {
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_human_detection);

        textureView = findViewById(R.id.videoTex);
        overlayView = findViewById(R.id.overlay);

        textureView.setSurfaceTextureListener(this);

        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Unable to load OpenCV!");
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully!");
            loadModel();
        }
    }

    private void loadModel() {
        try {
            MatOfByte modelBuffer = loadFileFromResource(R.raw.mobilenet_iter_73000);
            MatOfByte configBuffer = loadFileFromResource(R.raw.deploy);
            if (modelBuffer != null && configBuffer != null) {
                net = Dnn.readNetFromCaffe(configBuffer, modelBuffer);
                Log.d("HumanDetection", "Network loaded successfully");
            } else {
                Log.e("HumanDetection", "Failed to load model files!");
            }
        } catch (Exception e) {
            Log.e("HumanDetection", "Error loading model: " + e);
        }
    }

    private MatOfByte loadFileFromResource(int id) throws IOException {
        InputStream is = getResources().openRawResource(id);
        byte[] buffer = new byte[is.available()];
        int read = is.read(buffer);
        is.close();
        return new MatOfByte(buffer);
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        codecManager = new DJICodecManager(this, surface, width, height);

        videoDataListener = (videoBuffer, size) -> {
            if (codecManager != null) {
                codecManager.sendDataToDecoder(videoBuffer, size);
            }
        };
        VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(videoDataListener);

        codecManager.enabledYuvData(true);
        codecManager.setYuvDataCallback(new DJICodecManager.YuvDataCallback() {
            @Override
            public void onYuvDataReceived(MediaFormat mediaFormat, ByteBuffer yuvFrame,
                                          int dataSize, int w, int h) {
                try {
                    Bitmap bmp = textureView.getBitmap();
                    if (bmp != null) {
                        Mat frame = new Mat();
                        org.opencv.android.Utils.bitmapToMat(bmp, frame);
                        processFrame(frame);
                        frame.release();
                    }
                } catch (Exception e) {
                    Log.e("HumanDetection", "onYuvDataReceived error: " + e);
                }
            }
        });
    }

    private void processFrame(Mat frame) {
        final int IN_WIDTH = 300;
        final int IN_HEIGHT = 300;
        final double IN_SCALE_FACTOR = 0.007843;
        final double MEAN_VAL = 127.5;
        final double THRESHOLD = 0.2;

        // 建立輸入 blob
        Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
                new Size(IN_WIDTH, IN_HEIGHT),
                new Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), false, false);

        net.setInput(blob);
        Mat detections = net.forward();

        Log.d("HumanDetection", "Detections shape: " + detections.size());

        int cols = frame.cols();
        int rows = frame.rows();
        detections = detections.reshape(1, (int)detections.total() / 7);

        List<Rect> rects = new ArrayList<>();
        boolean danger = false;

        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > THRESHOLD) {
                int classId = (int)detections.get(i, 1)[0];
                String label = classNames[classId];

                if ("person".equals(label)) {
                    int left   = (int)(detections.get(i, 3)[0] * cols);
                    int top    = (int)(detections.get(i, 4)[0] * rows);
                    int right  = (int)(detections.get(i, 5)[0] * cols);
                    int bottom = (int)(detections.get(i, 6)[0] * rows);

                    rects.add(new Rect(left, top, right - left, bottom - top));
                    danger = true;

                    Log.d("HumanDetection", "Detection[" + i + "]: classId=" + classId + " conf=" + confidence);
                }
            }
        }

        // ✅ 讓 lambda 可以用（必須 final 或 effectively final）
        final List<Rect> finalRects = new ArrayList<>(rects);
        final boolean finalDanger = danger;

        runOnUiThread(() -> overlayView.updateDetections(finalRects, finalDanger));
    }


    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        if (videoDataListener != null) {
            VideoFeeder.getInstance().getPrimaryVideoFeed().removeVideoDataListener(videoDataListener);
        }
        if (codecManager != null) {
            codecManager.destroyCodec();
            codecManager = null;
        }
        return true;
    }
    @Override public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
}
