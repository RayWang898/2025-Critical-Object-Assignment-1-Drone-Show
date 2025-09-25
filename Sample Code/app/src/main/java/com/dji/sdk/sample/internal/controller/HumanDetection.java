//!!rrr
package com.dji.sdk.sample.internal.controller;

import android.graphics.SurfaceTexture;
import android.media.MediaFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.dji.sdk.sample.R;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.OpenCVLoader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import dji.sdk.codec.DJICodecManager;
import dji.sdk.camera.VideoFeeder;

public class HumanDetection extends AppCompatActivity implements TextureView.SurfaceTextureListener {

    private static final String TAG = "HumanDetection";

    private TextureView textureView;
    private DJICodecManager codecManager;
    private VideoFeeder.VideoDataListener videoDataListener;

    private Net net;  // MobileNet SSD

    // coco-like classes (caffe version)
    private static final String[] classNames = {"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_human_detection);

        // OpenCV 初始化
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Unable to load OpenCV!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
            return;
        } else {
            Log.d(TAG, "OpenCV loaded successfully!");
        }

        // 載入 DNN 模型
        MatOfByte modelBuffer = loadFileFromResource(R.raw.mobilenet_iter_73000);
        MatOfByte configBuffer = loadFileFromResource(R.raw.deploy);
        if (modelBuffer == null || configBuffer == null) {
            Log.e(TAG, "Failed to load DNN model");
        } else {
            net = Dnn.readNetFromCaffe(configBuffer, modelBuffer);
            Log.d(TAG, "Network loaded successfully");
        }

        // TextureView 綁定
        textureView = findViewById(R.id.videoTex);
        textureView.setSurfaceTextureListener(this);
    }

    private MatOfByte loadFileFromResource(int id) {
        try {
            InputStream is = getResources().openRawResource(id);
            byte[] buffer = new byte[is.available()];
            int bytesRead = is.read(buffer);
            is.close();
            return new MatOfByte(buffer);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // ---- TextureView callback ----
    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.d(TAG, "Surface ready, init codecManager");
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
                // 轉換成 Mat
                Mat frame = yuvToMat(yuvFrame, w, h);
                Log.d(TAG, "Frame captured: " + frame.cols() + "x" + frame.rows());

                // 物件偵測
                boolean detected = detectHuman(frame);

                // (TODO: 如果要顯示在畫面上，需要再 overlay 到 Surface 或另開 View)
            }
        });
    }

    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int w, int h) {}
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

    private Mat yuvToMat(ByteBuffer yuvFrame, int w, int h) {
        byte[] yuvBytes = new byte[yuvFrame.remaining()];
        yuvFrame.get(yuvBytes);

        Mat yuvMat = new Mat(h + h / 2, w, CvType.CV_8UC1);
        yuvMat.put(0, 0, yuvBytes);

        Mat bgrMat = new Mat();
        Imgproc.cvtColor(yuvMat, bgrMat, Imgproc.COLOR_YUV2BGR_I420);
        return bgrMat;
    }

    // ---- DNN 人偵測 ----
    private boolean detectHuman(Mat frame) {
        if (net == null) {
            Log.e(TAG, "DNN not loaded");
            return false;
        }

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);

        Mat blob = Dnn.blobFromImage(frame, 0.007843,
                new Size(300, 300),
                new Scalar(127.5, 127.5, 127.5),
                false, false);

        net.setInput(blob);
        Mat detections = net.forward();

        int cols = frame.cols();
        int rows = frame.rows();
        detections = detections.reshape(1, (int) detections.total() / 7);

        boolean found = false;

        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            int classId = (int) detections.get(i, 1)[0];


            if (confidence > 0.3) {
                int left   = (int)(detections.get(i, 3)[0] * cols);
                int top    = (int)(detections.get(i, 4)[0] * rows);
                int right  = (int)(detections.get(i, 5)[0] * cols);
                int bottom = (int)(detections.get(i, 6)[0] * rows);

                if (classId == 15) { // person
                    found = true;
                    Log.d(TAG, "Human Detected! " +  " conf=" + confidence);
                }
            }
        }
        return found;
    }
}
