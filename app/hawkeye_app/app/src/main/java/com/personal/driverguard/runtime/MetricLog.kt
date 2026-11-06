package com.personal.driverguard.runtime

import android.content.Context
import com.personal.driverguard.domain.engine.EngineSnapshot
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 개발 화면에서 사용자가 명시적으로 시작할 때만 쓰는 수치 로그(CSV).
 * 프레임별 지표 숫자만 남기고 영상·얼굴 이미지·랜드마크 좌표는 남기지 않는다. 앱 내부 저장소에만 쓰며 전송하지 않는다.
 */
class MetricLog(context: Context) {
    private val dir = File(context.filesDir, "metric_logs")
    private var writer: BufferedWriter? = null

    @Volatile
    var currentFile: File? = null
        private set

    val isRecording: Boolean get() = writer != null

    fun files(): List<File> = dir.listFiles()?.sortedByDescending { it.lastModified() }.orEmpty()

    @Synchronized
    fun start(): File {
        stop()
        dir.mkdirs()
        val file = File(dir, "metrics_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".csv")
        writer = file.bufferedWriter().also { it.appendLine(HEADER) }
        currentFile = file
        return file
    }

    @Synchronized
    fun append(s: EngineSnapshot) {
        val w = writer ?: return
        fun d(v: Double?) = v?.let { String.format(Locale.US, "%.4f", it) } ?: ""
        w.appendLine(
            listOf(
                s.timestampMs.toString(), s.faceQuality.name, s.eyeQuality.name, s.mouthQuality.name, s.poseQuality.name,
                d(s.leftEar), d(s.rightEar), d(s.filteredLeftEar), d(s.filteredRightEar), s.eyeState.name, s.eyeClosedMs.toString(),
                d(s.closedRatio.ratio), d(s.closedRatio.coverage), s.closedRatio.status.name,
                d(s.mouthRatio), d(s.yawDeg), d(s.pitchDeg), d(s.rollDeg), s.headAwayMs.toString(),
            ).joinToString(","),
        )
    }

    @Synchronized
    fun stop() {
        writer?.runCatching { flush(); close() }
        writer = null
        currentFile = null
    }

    @Synchronized
    fun deleteAll(): Int {
        stop()
        return files().count { it.delete() }
    }

    private companion object {
        const val HEADER = "t_ms,face_q,eye_q,mouth_q,pose_q,ear_l,ear_r,ear_l_f,ear_r_f,eye_state,eye_closed_ms," +
            "closed_ratio,coverage,ratio_status,mouth_ratio,yaw,pitch,roll,head_away_ms"
    }
}
