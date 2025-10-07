//!!rrr
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
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.products.Aircraft;

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

    // 最新偵測的水平中心 x (0~1)，找不到人時置中（0.5）
    private volatile float latestXNorm = 0.5f;
    // 送桿定時器（Virtual Stick 必須週期送出）
    private final Handler vsHandler = new Handler(Looper.getMainLooper());
    private final int VS_PERIOD_MS = 100; // 10 Hz
    private final Runnable vsLoop = new Runnable() {
        @Override public void run() {
            if (flightController != null && vsEnabled && followHuman) {
                // 將人臉（人）中心偏差換成 yaw 角速度
                float error = latestXNorm - 0.5f; // 左負右正
                float K = 60f;                    // 簡單比例增益
                float yawRate = K * error;        // deg/s
                yawRate = Math.max(-25f, Math.min(25f, yawRate)); // 安全限幅

                FlightControlData ctrl = new FlightControlData(
                        0f,  // pitch velocity
                        0f,  // roll velocity
                        yawRate, // yaw angular velocity (deg/s)
                        0f   // vertical velocity
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

        // 取得 FlightController
        Aircraft ac = (Aircraft) DJISampleApplication.getProductInstance();
        if (ac != null) {
            flightController = ac.getFlightController();
        }

        // OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Unable to load OpenCV!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
            return;
        } else {
            Log.d(TAG, "OpenCV loaded successfully!");
        }

        // 載入 DNN 模型（raw 資源）
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

        // 啟動 VS 送桿循環（先啟、但只有 followHuman && vsEnabled 才會送）
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

    // --- 任務：起飛 + 開啟 VS ---
    private void armed() {
        if (flightController == null) return;

        flightController.startTakeoff(takeoffError -> {
            if (takeoffError == null) {
                Log.d(TAG, "Armed: takeoff started");
                ToastUtils.setResultToToast("Armed: takeoff started");

                // 設定 VS 參數並啟用
                setupVirtualStick(true);

            } else {
                Log.e(TAG, "Armed: takeoff failed: " + takeoffError.getDescription());
                ToastUtils.setResultToToast("Armed: takeoff failed");
            }
        });
    }

    // --- 任務：降落（含 5s 後 confirm） ---
    private void terminate() {
        if (flightController == null) return;

        flightController.startLanding(djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Terminate: landing started");
                ToastUtils.setResultToToast("Terminate: landing started");

                new Handler(Looper.getMainLooper()).postDelayed(() ->
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

    // --- 任務：短測試（起飛→關 VS→降落） ---
    private void test() {
        if (flightController == null) return;

        flightController.startTakeoff(djiError -> {
            if (djiError == null) {
                Log.d(TAG, "Test: Takeoff started");
                ToastUtils.setResultToToast("Test: Takeoff started");

                new Handler(Looper.getMainLooper()).postDelayed(() -> {
                    // 確保關閉 VS 再降落
                    setupVirtualStick(false);

                    flightController.startLanding(landingError -> {
                        if (landingError == null) {
                            Log.d(TAG, "Test: landing started");
                            ToastUtils.setResultToToast("Test: landing started");
                            new Handler(Looper.getMainLooper()).postDelayed(() ->
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

    // --- Virtual Stick 設定/啟用 ---
    private void setupVirtualStick(boolean enable) {
        if (flightController == null) return;

        // 座標/控制模式
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

        yuvCallback = (MediaFormat mediaFormat, ByteBuffer yuvFrame, int dataSize, int w, int h) -> {
            // 轉成 Mat (I420 -> BGR)
            Mat frame = yuvToMat(yuvFrame, w, h);
            Log.d(TAG, "Frame captured: " + frame.cols() + "x" + frame.rows());

            // 人形偵測；若跟隨啟用，更新 latestXNorm 給 VS 週期器
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
        // 若有需要重複讀取，可 yuvFrame.rewind();

        Mat yuvMat = new Mat(h + h / 2, w, CvType.CV_8UC1);
        yuvMat.put(0, 0, yuvBytes);

        Mat bgrMat = new Mat();
        Imgproc.cvtColor(yuvMat, bgrMat, Imgproc.COLOR_YUV2BGR_I420);
        return bgrMat;
    }

    // ---- 偵測並更新 VS 目標 ----
    private void detectAndUpdate(Mat frameBGR) {
        if (net == null) return;

        Imgproc.cvtColor(frameBGR, frameBGR, Imgproc.COLOR_BGR2RGB);

        Mat blob = Dnn.blobFromImage(frameBGR, 0.007843,
                new Size(300, 300),
                new Scalar(127.5, 127.5, 127.5),
                false, false);

        net.setInput(blob);
        Mat detections = net.forward();

        int cols = frameBGR.cols();
        int rows = frameBGR.rows();
        detections = detections.reshape(1, (int) detections.total() / 7);

        double bestConf = 0.0;
        int bestLeft = 0, bestTop = 0, bestRight = 0, bestBottom = 0;

        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            int classId = (int) detections.get(i, 1)[0];

            if (confidence > 0.3 && classId == 15) { // 15 = person
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
            latestXNorm = Math.max(0f, Math.min(1f, (float)xCenter / (float)cols));

            // 顯示偵測結果
            Log.d(TAG, String.format("Human Detected! conf=%.2f, xNorm=%.2f", bestConf, latestXNorm));
            ToastUtils.setResultToToast(String.format("Human Detected! conf=%.2f", bestConf));
        } else {
            latestXNorm = 0.5f; // 沒人時歸中
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
