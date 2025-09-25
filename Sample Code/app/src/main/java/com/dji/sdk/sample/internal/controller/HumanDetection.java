//!!rrr
package com.dji.sdk.sample.internal.controller;

import android.graphics.SurfaceTexture;
import android.media.MediaFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.dji.sdk.sample.R;
import com.dji.sdk.sample.internal.utils.ToastUtils;

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

import dji.common.flightcontroller.virtualstick.FlightControlData;
import dji.common.flightcontroller.virtualstick.FlightCoordinateSystem;
import dji.common.flightcontroller.virtualstick.RollPitchControlMode;
import dji.common.flightcontroller.virtualstick.YawControlMode;
import dji.common.flightcontroller.virtualstick.VerticalControlMode;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.products.Aircraft;


public class HumanDetection extends AppCompatActivity implements TextureView.SurfaceTextureListener {

    private static final String TAG = "HumanDetection";
    private TextureView textureView;
    private DJICodecManager codecManager;
    private VideoFeeder.VideoDataListener videoDataListener;
    private Net net;  // MobileNet SSD

    // coco-like classes (caffe version) object detection list
    private static final String[] classNames = {"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};
    private FlightController flightController;
    private boolean followHuman = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_human_detection);

        // initiate flight controller
        Aircraft ac = (Aircraft) DJISampleApplication.getProductInstance();
        if (ac != null) {
            flightController = ac.getFlightController();
        }

        // OpenCV initiation
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Unable to load OpenCV!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
            return;
        } else {
            Log.d(TAG, "OpenCV loaded successfully!");
        }

        // load DNN model
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

    // drone behavior functions
    private void armed() {
        if (flightController == null) return;

        flightController.startTakeoff(takeoffError -> {
            if (takeoffError == null) {
                Log.d(TAG, "Armed: takeoff started");
                ToastUtils.setResultToToast("Armed: takeoff started");

                setupVirtualStick();

                // After 30 seconds, start the landing procedure
                new android.os.Handler().postDelayed(() -> {
                    flightController.startLanding(landingError -> {
                        if (landingError == null) {
                            Log.d(TAG, "Armed: landing started after 30s");
                            ToastUtils.setResultToToast("Armed: landing started after 30s");

                            // If the aircraft stays hovering (landing protection active),
                            // confirm landing after 5 seconds to force touchdown
                            new android.os.Handler().postDelayed(() -> {
                                flightController.confirmLanding(confirmError -> {
                                    if (confirmError == null) {
                                        Log.d(TAG, "Armed: landing confirmed, forcing touchdown");
                                        ToastUtils.setResultToToast("Armed: landing confirmed, forcing touchdown");
                                    } else {
                                        Log.e(TAG, "Armed: confirm landing failed: " + confirmError.getDescription());
                                        ToastUtils.setResultToToast("Armed: confirm landing failed");
                                    }
                                });
                            }, 5000);

                        } else {
                            Log.e(TAG, "Armed: landing failed: " + landingError.getDescription());
                            ToastUtils.setResultToToast("Armed: landing failed");
                        }
                    });
                }, 30000); // 30 seconds delay before landing

            } else {
                Log.e(TAG, "Armed: takeoff failed: " + takeoffError.getDescription());
                ToastUtils.setResultToToast("Armed: takeoff failed");
            }
        });
    }

    private void terminate() {
        if (flightController == null) return;

        //initiate landing
        flightController.startLanding(djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Terminate: landing started");
                ToastUtils.setResultToToast("Terminate: landing started");
                // if the drone didn't detect flat surface and didnt land 5s after, confirm and force landing
                new android.os.Handler().postDelayed(() -> {
                    flightController.confirmLanding(err -> {
                        if (err == null) {
                            Log.d(TAG, "Terminate: Landing confirmed, forcing touchdown");
                            ToastUtils.setResultToToast("Terminate: Landing confirmed, forcing touchdown");
                        } else {
                            Log.e(TAG, "Terminate: Confirm landing failed: " + err.getDescription());
                            ToastUtils.setResultToToast("Terminate: Confirm landing failed");
                        }
                    });
                }, 5000);

            } else {
                Log.e(TAG, "Terminate: Landing failed: " + djiError.getDescription());
                ToastUtils.setResultToToast("Terminate: Landing failed");
            }
        });
    }

    private void test(){
        if (flightController == null) return;

        flightController.startTakeoff(djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Test: Takeoff started");
                ToastUtils.setResultToToast("Test: Takeoff started");

                // wait for 6s, make sure the drone is flying still in 1.2m height and entered flying mode
                new android.os.Handler().postDelayed(() -> {

                    //make sure virtual stick is turned off(so that it doesn't yaw)
                    flightController.setVirtualStickModeEnabled(false, null);

                    //initiate landing
                    flightController.startLanding(landingError -> {
                        if (landingError == null) {
                            Log.d(TAG, "Test: landing started");
                            ToastUtils.setResultToToast("Test: landing started");
                            // if the drone didn't detect flat surface and didnt land 5s after, confirm and force landing
                            new android.os.Handler().postDelayed(() -> {
                                flightController.confirmLanding(err -> {
                                    if (err == null) {
                                        Log.d(TAG, "Test: Landing confirmed, forcing touchdown");
                                        ToastUtils.setResultToToast("Test: Landing confirmed, forcing touchdown");
                                    } else {
                                        Log.e(TAG, "Test: Confirm landing failed: " + err.getDescription());
                                        ToastUtils.setResultToToast("Test: Confirm landing failed");
                                    }
                                });
                            }, 5000);

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

    private void setupVirtualStick() {
        flightController.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY);
        flightController.setRollPitchControlMode(RollPitchControlMode.VELOCITY);
        flightController.setYawControlMode(YawControlMode.ANGULAR_VELOCITY);
        flightController.setVerticalControlMode(VerticalControlMode.VELOCITY);

        flightController.setVirtualStickModeEnabled(true, djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Virtual stick enabled");
            }
        });
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

    //yaw control function
    private void facePerson(float xNorm) {
        if (flightController == null) return;

        float error = xNorm - 0.5f;   // 左負右正
        float K = 60f;                // 比例增益
        float yawRate = K * error;    // deg/s
        yawRate = Math.max(-25f, Math.min(25f, yawRate));

        FlightControlData ctrl = new FlightControlData(
                0f,   // pitch
                0f,   // roll
                yawRate, // yaw 角速度
                0f    // throttle
        );
        flightController.sendVirtualStickFlightControlData(ctrl, null);
    }


    // ---- DNN human detection ----
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
                    Log.d(TAG, "Human Detected! conf=" + confidence);
                    ToastUtils.setResultToToast("Human Detected! conf=" + confidence);

                    if (followHuman) {
                        int xCenter = (left + right) / 2;
                        float xNorm = (float)xCenter / (float)cols;
                        facePerson(xNorm);
                    }
                }

            }
        }
        return found;
    }

    public void onArmedClick(View v) {
        armed();
        followHuman = true;
    }

    public void onLandClick(View v) {
        terminate();
        followHuman = false;
    }

    public void onTestClick(View v) {
        test();
        followHuman = false;
    }

}
