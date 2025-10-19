package com.dji.sdk.sample.internal.controller;

import android.graphics.SurfaceTexture;
import android.media.MediaFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.dji.sdk.sample.R;
import com.dji.sdk.sample.internal.utils.ToastUtils;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import dji.common.flightcontroller.virtualstick.FlightControlData;
import dji.common.flightcontroller.virtualstick.FlightCoordinateSystem;
import dji.common.flightcontroller.virtualstick.RollPitchControlMode;
import dji.common.flightcontroller.virtualstick.VerticalControlMode;
import dji.common.flightcontroller.virtualstick.YawControlMode;
import dji.common.gimbal.GimbalMode;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.products.Aircraft;

import dji.sdk.gimbal.Gimbal;
import dji.common.gimbal.GimbalMode;
import dji.common.gimbal.Rotation;
import dji.common.gimbal.RotationMode;

public class HumanDetection extends AppCompatActivity implements TextureView.SurfaceTextureListener {
    private static final String TAG = "HumanDetection";
    private TextureView textureView;
    private DJICodecManager codecManager;
    private VideoFeeder.VideoDataListener videoDataListener;
    private DJICodecManager.YuvDataCallback yuvCallback;
    private Net net; // MobileNet-SSD (Caffe)

    // Caffe MobileNet-SSD class list
    private static final String[] classNames = {"background",
            "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
            "cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};

    private FlightController flightController;

    // --- Follow / control state ---
    private volatile boolean followHuman = false;
    private volatile boolean vsEnabled = false;

    // The latest detected horizontal center x (range 0–1); if no person is found, it defaults to the center (0.5).
    private volatile float latestXNorm = 0.5f;
    //Virtual stick command send out timer (Virtual Stick commands must be sent periodically)
    private final Handler vsHandler = new Handler(Looper.getMainLooper());
    private final int VS_PERIOD_MS = 100; // 10 Hz
    private Gimbal gimbal;
    private final Runnable vsLoop = new Runnable() {
        @Override public void run() {
            if (flightController != null && vsEnabled && followHuman) {
                // Map the detected face center offset to a yaw angular velocity
                float error = latestXNorm - 0.5f; // left: negative; right: positive
                float K = 60f;
                float yawRate = K * error;        // deg/s

                FlightControlData ctrl = new FlightControlData(
                        0f,  // pitch velocity (x)
                        0f,  // roll velocity (y)
                        yawRate, // yaw angular velocity (deg/s)
                        0f   // vertical velocity (z)
                );
                flightController.sendVirtualStickFlightControlData(ctrl, null);
            }
            vsHandler.postDelayed(this, VS_PERIOD_MS);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_human_detection);

        // OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Unable to load OpenCV!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
            return;
        } else {
            Log.d(TAG, "OpenCV loaded successfully!");
        }

        // Loading DNN model (Deep Neural Network model) -->for OpenCV object recognition
        MatOfByte modelBuffer = loadFileFromResource(R.raw.mobilenet_iter_73000);
        MatOfByte configBuffer = loadFileFromResource(R.raw.deploy);
        if (modelBuffer == null || configBuffer == null) {
            Log.e(TAG, "Failed to load DNN model");
        } else {
            net = Dnn.readNetFromCaffe(configBuffer, modelBuffer);
            Log.d(TAG, "Network loaded successfully");
        }

        // TextureView
        textureView = findViewById(R.id.videoTex);
        textureView.setSurfaceTextureListener(this);

        // initiate virtual stick handler process (buy only operate in followHuman && vsEnabled)
        vsHandler.post(vsLoop);
    }

    private MatOfByte loadFileFromResource(int id) {
        try (InputStream is = getResources().openRawResource(id)) {
            byte[] buffer = new byte[is.available()];
            int bytesRead = is.read(buffer);
            if (bytesRead <= 0) return null;
            return new MatOfByte(buffer);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // --- mission: armed--> takeoff + initiate VS ---
    private void armed() {
        if (flightController == null) return;

        flightController.startTakeoff(takeoffError -> {
            if (takeoffError == null) {
                Log.d(TAG, "Armed: takeoff started");
                ToastUtils.setResultToToast("Armed: takeoff started");

                // delay 5s, turn on VS only after drone is hovering steadily (if initiate too early--> VS data wont send in)
                new android.os.Handler().postDelayed(() -> {
                    setupVirtualStick(true);
                }, 5000);

            } else {
                Log.e(TAG, "Armed: takeoff failed: " + takeoffError.getDescription());
                ToastUtils.setResultToToast("Armed: takeoff failed");
            }
        });
    }

    // --- mission: landing ---
    private void terminate() {
        if (flightController == null) return;
        // start landing: decrease flying height and land if flat surface is detected
        flightController.startLanding(djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Terminate: landing started");
                ToastUtils.setResultToToast("Terminate: landing started");

                new Handler(Looper.getMainLooper()).postDelayed(() ->
                        // confirm landing: force landing if yet landed 5s after start landing (landing even if no flat surface is detected)
                        flightController.confirmLanding(err -> {
                            if (err == null) {
                                Log.d(TAG, "Terminate: Landing confirmed");
                                ToastUtils.setResultToToast("Terminate: Landing confirmed");
                            } else {
                                Log.e(TAG, "Terminate: Confirm landing failed: " + err.getDescription());
                                ToastUtils.setResultToToast("Terminate: Confirm landing failed");
                            }
                        }), 5000);

            } else {
                Log.e(TAG, "Terminate: Landing failed: " + djiError.getDescription());
                ToastUtils.setResultToToast("Terminate: Landing failed");
            }
        });
    }

    // --- mission: simple fly test (take off and landing when VS is turned off)
    private void test() {
        if (flightController == null) return;

        flightController.startTakeoff(djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Test: Takeoff started");
                ToastUtils.setResultToToast("Test: Takeoff started");

                new Handler(Looper.getMainLooper()).postDelayed(() -> {
                    setupVirtualStick(false);
                    // start landing: decrease flying height and land if flat surface is detected
                    flightController.startLanding(landingError -> {
                        if (landingError == null) {
                            Log.d(TAG, "Test: landing started");
                            ToastUtils.setResultToToast("Test: landing started");
                            new Handler(Looper.getMainLooper()).postDelayed(() ->
                                    // confirm landing: force landing if yet landed 5s after start landing (landing even if no flat surface is detected)
                                    flightController.confirmLanding(err -> {
                                        if (err == null) {
                                            Log.d(TAG, "Test: Landing confirmed");
                                            ToastUtils.setResultToToast("Test: Landing confirmed");
                                        } else {
                                            Log.e(TAG, "Test: Confirm landing failed: " + err.getDescription());
                                            ToastUtils.setResultToToast("Test: Confirm landing failed");
                                        }
                                    }), 5000);
                        } else {
                            Log.e(TAG, "Test: Landing failed: " + landingError.getDescription());
                            ToastUtils.setResultToToast("Test: Landing failed");
                        }
                    });
                }, 6000);

            } else {
                Log.e(TAG, "Test: Takeoff failed - " + djiError.getDescription());
                ToastUtils.setResultToToast("Test: Takeoff failed");
            }
        });
    }

    // --- Virtual Stick setting/initiation ---
    private void setupVirtualStick(boolean enable) {
        if (flightController == null) return;

        // Coordinate / Control Mode
        flightController.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY);
        flightController.setRollPitchControlMode(RollPitchControlMode.VELOCITY);
        flightController.setYawControlMode(YawControlMode.ANGULAR_VELOCITY);
        flightController.setVerticalControlMode(VerticalControlMode.VELOCITY);

        flightController.setVirtualStickModeEnabled(enable, djiError -> {
            if (djiError == null) {
                vsEnabled = enable;
                Log.d(TAG, "Virtual stick " + (enable ? "enabled" : "disabled"));
            } else {
                vsEnabled = false;
                Log.e(TAG, "Virtual stick set failed: " + djiError.getDescription());
            }
        });
    }

    // ---- @@! TextureView callback (Main part dealing with sending data to openCV for recognition) ----
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

        yuvCallback = (MediaFormat mediaFormat, ByteBuffer yuvFrame, int dataSize, int w, int h) -> {
            // turn into Mat format (I420 -> BGR) --> Mat is OpenCV accepted data format
            Mat frame = yuvToMat(yuvFrame, w, h);
            Log.d(TAG, "Frame captured: " + frame.cols() + "x" + frame.rows());

            // Human detection; if tracking is enabled, update latestXNorm for the VS scheduler
            detectAndUpdate(frame);
        };
        codecManager.setYuvDataCallback(yuvCallback);
    }

    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int w, int h) {}

    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        if (videoDataListener != null) {
            VideoFeeder.getInstance().getPrimaryVideoFeed().removeVideoDataListener(videoDataListener);
            videoDataListener = null;
        }
        if (codecManager != null) {
            codecManager.enabledYuvData(false);
            codecManager.setYuvDataCallback(null);
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

    // ---- detect and renew VS target ----
    private void detectAndUpdate(Mat frameBGR) {
        if (net == null) return;

        // Convert for DNN
        Imgproc.cvtColor(frameBGR, frameBGR, Imgproc.COLOR_BGR2RGB);
        Mat blob = Dnn.blobFromImage(
                frameBGR,
                0.007843,                    // scale
                new Size(300, 300),          // input size for MobileNet-SSD
                new Scalar(127.5,127.5,127.5),
                false,                       // swapRB
                false                        // crop
        );

        Mat detections = new Mat();
        try {
            net.setInput(blob);
            detections = net.forward();

            final int cols = frameBGR.cols();
            final int rows = frameBGR.rows();

            detections = detections.reshape(1, (int) detections.total() / 7);

            double bestConf = 0.0;
            int bestLeft = 0, bestTop = 0, bestRight = 0, bestBottom = 0;

            for (int i = 0; i < detections.rows(); ++i) {
                double confidence = detections.get(i, 2)[0];
                int classId = (int) detections.get(i, 1)[0];

                if (confidence > 0.3 && classId == 15) { // 15 = "person"
                    int left   = (int)(detections.get(i, 3)[0] * cols);
                    int top    = (int)(detections.get(i, 4)[0] * rows);
                    int right  = (int)(detections.get(i, 5)[0] * cols);
                    int bottom = (int)(detections.get(i, 6)[0] * rows);

                    if (confidence > bestConf) {
                        bestConf = confidence;
                        bestLeft = left; bestTop = top; bestRight = right; bestBottom = bottom;
                    }
                }
            }

            if (bestConf > 0.3) {
                int xCenter = (bestLeft + bestRight) / 2;
                latestXNorm = Math.max(0f, Math.min(1f, (float) xCenter / (float) cols));

                // --- Gimbal yaw micro-adjust (relative angle) ---
                if (gimbal != null) {
                    float err = latestXNorm - 0.5f;          // [-0.5, +0.5]
                    float maxGimbalYawDeg = 15f;             // small, smooth correction
                    float deltaYawDeg = err * 2f * maxGimbalYawDeg;

                    // clamp to a safe micro step each frame
                    if (deltaYawDeg > 5f)  deltaYawDeg = 5f;
                    if (deltaYawDeg < -5f) deltaYawDeg = -5f;

                    Rotation r = new Rotation.Builder()
                            .mode(RotationMode.RELATIVE_ANGLE)
                            .yaw(deltaYawDeg)
                            .build();

                    gimbal.rotate(r, djiError -> {
                        if (djiError != null) {
                            Log.w(TAG, "Gimbal rotate failed: " + djiError.getDescription());
                        }
                    });
                }

                Log.d(TAG, String.format("Human Detected conf=%.2f, xNorm=%.2f", bestConf, latestXNorm));
                ToastUtils.setResultToToast(String.format("Human Detected! conf=%.2f", bestConf));
            } else {
                latestXNorm = 0.5f; // center when not found
            }
        } finally {
            blob.release();
            detections.release();
        }
    }



    // --- UI onClick ---
    public void onArmedClick(View v) {
        armed();
        followHuman = true;
    }

    public void onLandClick(View v) {
        followHuman = false;
        setupVirtualStick(false);
        terminate();
    }

    public void onTestClick(View v) {
        followHuman = false;
        test();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        followHuman = false;
        setupVirtualStick(false);
        vsHandler.removeCallbacksAndMessages(null);
    }
}
