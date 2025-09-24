//!!rrr
package com.dji.sdk.sample.internal.controller;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;

import androidx.appcompat.app.AppCompatActivity;

import com.dji.sdk.sample.R;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.android.Utils;

import dji.sdk.codec.DJICodecManager;
import dji.sdk.camera.VideoFeeder;

public class HumanDetection extends AppCompatActivity implements TextureView.SurfaceTextureListener {

    private TextureView textureView;
    private DJICodecManager codecManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_human_detection);

        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Unable to load OpenCV!");
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully!");
        }

        textureView = findViewById(R.id.videoTex);
        textureView.setSurfaceTextureListener(this);
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.d("HumanDetection", "Surface ready, init codecManager");
        codecManager = new DJICodecManager(this, surface, width, height);

        // 綁定影像流（送進 codecManager，才能顯示畫面）
        VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener((videoBuffer, size) -> {
            if (codecManager != null) {
                codecManager.sendDataToDecoder(videoBuffer, size);
            }
        });
    }

    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        if (codecManager != null) {
            codecManager.destroyCodec();
            codecManager = null;
        }
        return true;
    }

    // ⚡ 每一幀更新時觸發：抓 Bitmap → 轉成 Mat
    private long lastTime = 0;

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        long now = System.currentTimeMillis();
        if (now - lastTime < 200) return; // 每 200ms 抓一次 (~5 FPS)
        lastTime = now;

        Bitmap bmp = textureView.getBitmap(640, 360); // 直接縮小
        if (bmp != null) {
            Mat frame = new Mat();
            Utils.bitmapToMat(bmp, frame);
            Log.d("HumanDetection", "Frame captured: " + frame.size());
            bmp.recycle();
        }
    }


    private boolean detectHuman(Mat frame) {
        // TODO: 這裡寫真正的 OpenCV 偵測
        return false;
    }
}
