package com.personal.driverguard.views.components

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.foundation.layout.size
import com.personal.driverguard.views.theme.sizes
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import com.personal.driverguard.R
import com.personal.driverguard.views.theme.spacing
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.personal.driverguard.analysis.FaceOverlay
import com.personal.driverguard.domain.analysis.geometry.FaceLandmarkIndices
import com.personal.driverguard.views.theme.status

/**
 * 분석 이미지 좌표 → 화면 미리보기 좌표 변환.
 * 미리보기는 PreviewView FILL_CENTER(가운데 채움·잘림)이고 전면 카메라라 거울상이다.
 * 분석 좌표는 비미러이므로 화면에 그릴 때만 x 를 뒤집는다(모델 입력은 그대로 둔다).
 */
class PreviewMapping(viewW: Float, viewH: Float, imageW: Int, imageH: Int, private val mirror: Boolean = true) {
    private val scale = maxOf(viewW / imageW, viewH / imageH)
    private val offX = (viewW - imageW * scale) / 2f
    private val offY = (viewH - imageH * scale) / 2f
    private val width = viewW

    fun map(x: Float, y: Float): Offset {
        val sx = offX + x * scale
        return Offset(if (mirror) width - sx else sx, offY + y * scale)
    }
}

/**
 * 얼굴 테두리(기본) 또는 개발용 랜드마크·인덱스·자세 축 오버레이.
 * [debug] 면 전체 점, EAR·입·자세 점 강조, [showIndices] 면 강조 점의 인덱스 번호, 재투영한 x·y·z 축을 그린다.
 */
@Composable
fun FaceOverlayView(
    overlay: FaceOverlay?,
    color: Color,
    modifier: Modifier = Modifier,
    debug: Boolean = false,
    showIndices: Boolean = false,
) {
    val axisColors = listOf(MaterialTheme.status.dawn, MaterialTheme.status.clear, MaterialTheme.status.onDemo)
    val muted = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.35f)
    val labelPaint = remember { Paint(Paint.ANTI_ALIAS_FLAG) }
    Canvas(modifier) {
        val o = overlay ?: return@Canvas
        val points = o.points ?: return@Canvas
        val m = PreviewMapping(size.width, size.height, o.imageWidth, o.imageHeight)
        fun p(i: Int) = m.map(points[i * 2] * o.imageWidth, points[i * 2 + 1] * o.imageHeight)

        if (!debug) {
            drawFaceFrame(points, o, m, color)
            return@Canvas
        }
        val count = points.size / 2
        for (i in 0 until count) drawCircle(muted, radius = 1.2.dp.toPx(), center = p(i))
        val outline = Stroke(width = 1.5.dp.toPx())
        for (eye in listOf(FaceLandmarkIndices.RIGHT_EYE_EAR, FaceLandmarkIndices.LEFT_EYE_EAR)) {
            FaceLandmarkIndices.earOutline(eye).forEach { (a, b) -> drawLine(color, p(a), p(b), outline.width) }
        }
        for ((a, b) in FaceLandmarkIndices.MOUTH_INNER_VERTICAL_PAIRS) drawLine(color, p(a), p(b), outline.width)
        drawLine(color, p(FaceLandmarkIndices.MOUTH_RIGHT_CORNER), p(FaceLandmarkIndices.MOUTH_LEFT_CORNER), outline.width)
        FaceLandmarkIndices.POSE_POINTS.forEach { drawCircle(color, radius = 2.5.dp.toPx(), center = p(it)) }

        o.axes?.let { axes ->
            val origin = m.map(axes[0], axes[1])
            for (k in 0 until 3) {
                drawLine(axisColors[k], origin, m.map(axes[2 + k * 2], axes[3 + k * 2]), 3.dp.toPx())
            }
        }
        if (showIndices) {
            labelPaint.color = color.toArgb()
            labelPaint.textSize = 9.sp.toPx()
            FaceLandmarkIndices.HIGHLIGHTED.forEach { i ->
                val c = p(i)
                drawContext.canvas.nativeCanvas.drawText(i.toString(), c.x + 3.dp.toPx(), c.y - 3.dp.toPx(), labelPaint)
            }
        }
    }
}

private fun DrawScope.drawFaceFrame(points: FloatArray, o: FaceOverlay, m: PreviewMapping, color: Color) {
    var minX = Float.MAX_VALUE
    var minY = Float.MAX_VALUE
    var maxX = -Float.MAX_VALUE
    var maxY = -Float.MAX_VALUE
    for (i in 0 until points.size / 2) {
        val c = m.map(points[i * 2] * o.imageWidth, points[i * 2 + 1] * o.imageHeight)
        minX = minOf(minX, c.x); maxX = maxOf(maxX, c.x)
        minY = minOf(minY, c.y); maxY = maxOf(maxY, c.y)
    }
    val pad = (maxX - minX) * 0.08f
    val topLeft = Offset(minX - pad, minY - pad)
    val box = Size(maxX - minX + pad * 2, maxY - minY + pad * 2)
    val radius = CornerRadius(box.minDimension * 0.18f)
    drawRoundRect(color, topLeft, box, radius, style = Stroke(width = 3.dp.toPx()))
}

/**
 * 카메라 영상 모서리의 '영상은 저장·전송하지 않음' 표시. 얼굴을 가리지 않도록 작게, 블러 없이 둔다.
 * 창이 작으면 글 대신 자물쇠 아이콘만 두고, 문장은 스크린 리더로 읽힌다.
 */
@Composable
fun PrivacyChip(modifier: Modifier = Modifier, compact: Boolean = false) {
    val text = stringResource(R.string.camera_privacy_chip)
    if (compact) {
        Icon(
            Icons.Rounded.Lock,
            contentDescription = text,
            tint = MaterialTheme.colorScheme.onSurface,
            modifier = modifier
                .clip(CapsuleShape)
                .background(MaterialTheme.colorScheme.surface)
                .padding(MaterialTheme.spacing.xxs)
                .size(MaterialTheme.sizes.iconS),
        )
        return
    }
    Text(
        text,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onBackground,
        maxLines = 1,
        modifier = modifier
            .clip(CapsuleShape)
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = MaterialTheme.spacing.s, vertical = MaterialTheme.spacing.xxs),
    )
}
