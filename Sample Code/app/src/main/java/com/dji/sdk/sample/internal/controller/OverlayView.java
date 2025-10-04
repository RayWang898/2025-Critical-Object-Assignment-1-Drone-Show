package com.dji.sdk.sample.internal.controller;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {
    private final Paint rectPaint;
    private List<RectF> boxes = new ArrayList<>();

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        rectPaint = new Paint();
        rectPaint.setColor(0xFFFF0000); // 紅框
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(5f);
    }

    public void setBoxes(List<RectF> boxes) {
        this.boxes = boxes;
        invalidate(); // 觸發重繪
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (RectF box : boxes) {
            canvas.drawRect(box, rectPaint);
        }
    }
}
