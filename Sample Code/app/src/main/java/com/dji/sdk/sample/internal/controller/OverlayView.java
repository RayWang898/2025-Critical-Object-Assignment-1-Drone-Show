package com.dji.sdk.sample.internal.controller;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {
    private final Paint boxPaint;
    private final Paint textPaint;
    private final List<Rect> rects = new ArrayList<>();
    private boolean danger = false;

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);

        textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(80f);
        textPaint.setStyle(Paint.Style.FILL);
    }

    public void updateDetections(List<Rect> newRects, boolean dangerFlag) {
        rects.clear();
        rects.addAll(newRects);
        danger = dangerFlag;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (Rect r : rects) {
            canvas.drawRect(r, boxPaint);
        }

        if (danger) {
            String dangerText = "DANGER";
            float x = (getWidth() - textPaint.measureText(dangerText)) / 2;
            float y = getHeight() / 2f;
            canvas.drawText(dangerText, x, y, textPaint);
        }
    }
}
