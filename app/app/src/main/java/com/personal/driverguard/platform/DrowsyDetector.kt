package com.personal.driverguard.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.personal.driverguard.analysis.FaceModelAsset
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.analysis.landmark.MediaPipeFaceLandmarkExtractor
import hawkeye.drowsy.DrowsinessMonitor
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.session.CalibrationStatus
import hawkeye.drowsy.core.session.DriverState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicBoolean

/** 감지기가 앱에 알리는 일 */
enum class Detection {
    /** 졸음(눈 감김 1초 이상, 3초 이상, 또는 최근 60초의 눈 감김 비율이 높음) */
    DROWSY,

    /** 전방 주시 이탈(3초 이상, 또는 최근 30초 중 누적 10초) */
    LOOK_AWAY,

    /** 다시 정면을 봄 */
    LOOK_BACK,
}

/** 화면에 보여 줄 감지 상태 */
data class DetectorUi(
    val running: Boolean = false,
    /** 운전자의 평소 눈 크기·자세를 잡는 중(시작 후 60~180초). 이 동안 시선 판정은 기다린다. */
    val calibrating: Boolean = false,
    val faceVisible: Boolean = false,
    /** 눈 감김 정도 0(뜸)~1(감음). 모르면 null */
    val closure: Double? = null,
    /** 기준 방향에서 벗어난 좌우 각도(도). 모르면 null */
    val yawDeg: Double? = null,
    val error: String? = null,
)

/**
 * 카메라 프레임 → MediaPipe 얼굴 랜드마크 → 졸음 판정 라이브러리(ai/kotlin 의 DrowsinessMonitor) → [Detection].
 *
 * 라이브러리가 요구하는 조건(ai/kotlin/README.md):
 *  - FaceLandmarker 는 LIVE_STREAM, 얼굴 1개, 블렌드셰이프와 변환 행렬 출력을 켠다. 모델은 공개 face_landmarker.task.
 *  - 프레임은 똑바로 세워서 넘기고(rotationDegrees 를 MediaPipe 에 맡기지 않는다) 좌우 반전하지 않는다.
 *  - 시각은 엄격히 늘어나는 ms. 얼굴이 없는 결과도 빠짐없이 넣는다. 운전 한 번에 인스턴스 하나(새로 보정).
 *  - 속도(km/h)를 넘기면 저속에서는 경고를 막는다(시선·감김 20km/h, 졸음 50km/h 미만). 모르면 null.
 *
 * 한 번에 한 프레임만 처리한다(처리 중이면 새 프레임은 버린다). 영상과 사진은 저장하지 않는다.
 * 라이브러리가 예외로 멈추면 감지를 끄고 [ui] 의 error 로 알린다(관찰과 기록은 계속된다).
 */
class DrowsyDetector(context: Context, private val onDetection: (Detection) -> Unit) {
    private val appContext = context.applicationContext
    private val busy = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val rotate = Matrix()

    @Volatile private var landmarker: FaceLandmarker? = null
    @Volatile private var monitor: DrowsinessMonitor? = null
    @Volatile private var frameWidth = 0
    @Volatile private var frameHeight = 0
    private var lastSubmitMs = 0L
    private var lookingAway = false

    /** 차량 속도. 주행 기록의 위치에서 계산해 넣는다. */
    @Volatile var speedKmh: Double? = null

    private val _ui = MutableStateFlow(DetectorUi())
    val ui: StateFlow<DetectorUi> = _ui.asStateFlow()

    /** 모델을 읽고 감지를 준비한다(수백 ms 걸릴 수 있어 백그라운드에서 부른다). 실패하면 false. */
    fun start(): Boolean = try {
        val asset = FaceModelAsset.check(appContext)
        val sha = (asset as? ModelAssetStatus.Ready)?.sha256 ?: error("얼굴 랜드마크 모델 파일을 읽지 못했습니다")
        monitor = DrowsinessMonitor.open(sha)
        landmarker = FaceLandmarker.createFromOptions(
            appContext,
            FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(MediaPipeFaceLandmarkExtractor.MODEL_ASSET_PATH).setDelegate(Delegate.CPU).build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setOutputFaceBlendshapes(true)
                .setOutputFacialTransformationMatrixes(true)
                .setResultListener { result, _ -> onResult(result) }
                .setErrorListener { e -> Log.w(TAG, "MediaPipe 오류", e); busy.set(false) }
                .build(),
        )
        _ui.value = DetectorUi(running = true, calibrating = true)
        true
    } catch (e: Throwable) {
        Log.e(TAG, "졸음 감지를 시작하지 못했습니다", e)
        fail(e)
        false
    }

    /** 카메라 분석 스레드에서 프레임마다 부른다. ImageProxy 는 호출자가 닫는다. */
    fun analyze(image: ImageProxy) {
        val lm = landmarker ?: return
        if (closed.get()) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastSubmitMs < MIN_INTERVAL_MS) return
        // MediaPipe 가 결과를 돌려주지 않고 프레임을 버리면 busy 가 풀리지 않는다. 오래 걸리면 풀어 감지가 멈추지 않게 한다.
        if (busy.get() && now - lastSubmitMs > STUCK_MS) {
            Log.w(TAG, "결과가 ${now - lastSubmitMs}ms 동안 오지 않아 다음 프레임을 보냅니다")
            busy.set(false)
        }
        if (!busy.compareAndSet(false, true)) return
        try {
            val upright = upright(image)
            frameWidth = upright.width
            frameHeight = upright.height
            lastSubmitMs = now
            lm.detectAsync(BitmapImageBuilder(upright).build(), now)
        } catch (e: Throwable) {
            Log.w(TAG, "프레임을 넘기지 못했습니다", e)
            busy.set(false)
        }
    }

    /** 회전 정보대로 똑바로 세운 복사본(좌우 반전 없음). 프레임마다 새 Bitmap 이라 MediaPipe 가 쓰는 동안 바뀌지 않는다. */
    private fun upright(image: ImageProxy): Bitmap {
        val src = image.toBitmap()
        val degrees = image.imageInfo.rotationDegrees
        if (degrees % 360 == 0) return src
        rotate.reset()
        rotate.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, rotate, true)
    }

    private fun onResult(result: FaceLandmarkerResult) {
        try {
            if (closed.get()) return
            val m = monitor ?: return
            val state = m.updateFromMediaPipe(result, frameWidth, frameHeight, speedKmh) ?: return
            publish(state)
        } catch (e: Throwable) {
            // 라이브러리는 처리 중 예외가 나면 그 세션을 멈춘다. 감지를 끄고 알린다.
            Log.e(TAG, "졸음 판정이 멈췄습니다", e)
            fail(e)
        } finally {
            busy.set(false)
        }
    }

    private fun publish(state: DriverState) {
        val alerts = state.alerts.map { it.type }
        // 무엇이 언제 감지됐는지 확인할 수 있게 남긴다(영상·얼굴 정보는 없다)
        if (state.events.isNotEmpty() || alerts.isNotEmpty()) {
            Log.i(TAG, "t=${state.sessionMs}ms 이벤트=${state.events.map { "${it.type}/${it.phase}" }} 경고=$alerts 속도=${state.speedKmh} 보정=${state.calibration}")
        }
        if (alerts.any { it == EventType.SLEEP || it == EventType.MICROSLEEP || it == EventType.DROWSY }) {
            onDetection(Detection.DROWSY)
        } else if (alerts.any { it == EventType.GAZE_AWAY_LONG || it == EventType.VATS }) {
            lookingAway = true
            onDetection(Detection.LOOK_AWAY)
        }
        // 시선 이탈 경고 뒤 이탈이 끝나면 정면 복귀로 알린다
        if (lookingAway && !fallbackAway && !state.isActive(EventType.GAZE_AWAY_LONG) && !state.isActive(EventType.VATS)) {
            lookingAway = false
            onDetection(Detection.LOOK_BACK)
        }
        val f = state.output.state
        if (!f.poseValid || fallbackAway) fallbackGaze(f.face, f.poseValid, state.sessionMs, state.speedKmh)
        _ui.value = DetectorUi(
            running = true,
            calibrating = state.calibration == CalibrationStatus.CALIBRATING,
            faceVisible = f.face,
            closure = f.closure.takeIf { f.closureValid && it.isFinite() },
            // 라이브러리는 평소 자세 기준이 잡힌 뒤(시작 후 약 1분)에만 머리 방향을 준다.
            // 그 전에는 화면에 보여 줄 값만 따로 계산한다(판정에는 쓰지 않는다).
            yawDeg = f.yawRel.takeIf { f.poseValid && it.isFinite() } ?: provisionalYawDeg().takeIf { f.face },
        )
    }

    private fun fail(e: Throwable) {
        _ui.value = DetectorUi(error = e.message ?: e.javaClass.simpleName)
        release()
    }

    fun close() {
        closed.set(true)
        release()
        _ui.value = DetectorUi()
    }

    private fun release() {
        val lm = landmarker
        val m = monitor
        landmarker = null
        monitor = null
        runCatching { lm?.close() }
        runCatching { m?.close() }
    }

    /** MediaPipe 결과 하나를 라이브러리 입력으로 바꾼다(ai/kotlin/README.md 의 변환 함수). 유한하지 않은 값이 있으면 얼굴 없는 프레임으로 넣는다. */
    private fun DrowsinessMonitor.updateFromMediaPipe(result: FaceLandmarkerResult, imageWidth: Int, imageHeight: Int, speedKmh: Double?): DriverState? {
        val tMs = result.timestampMs()
        val faces = result.faceLandmarks()
        val shapes = result.faceBlendshapes().orElse(null)
        val matrices = result.facialTransformationMatrixes().orElse(null)
        if (faces.isEmpty() || shapes.isNullOrEmpty() || matrices.isNullOrEmpty()) return updateNoFace(tMs, imageWidth, imageHeight, speedKmh)
        val landmarks = faces[0]
        val x = FloatArray(landmarks.size) { landmarks[it].x() }
        val y = FloatArray(landmarks.size) { landmarks[it].y() }
        val names = shapes[0].map { it.categoryName() }
        val scores = FloatArray(shapes[0].size) { shapes[0][it].score() }
        val matrix = matrices[0]
        trackRawYaw(matrix)
        if (x.any { !it.isFinite() } || y.any { !it.isFinite() } || scores.any { !it.isFinite() } || matrix.any { !it.isFinite() }) {
            return updateNoFace(tMs, imageWidth, imageHeight, speedKmh)
        }
        return update(tMs, imageWidth, imageHeight, x, y, names, scores, matrix, speedKmh)
    }

    // ── 기준이 잡히기 전의 머리 방향(화면 표시용) ──
    private var rawYawDeg: Double? = null
    private var yawZeroSum = 0.0
    private var yawZeroCount = 0

    /**
     * 얼굴 변환 행렬(열 우선 4×4)에서 좌우 각도를 구한다. 라이브러리와 같은 식이다(yaw = atan2(m[8], m[10])).
     * 처음 [YAW_ZERO_FRAMES] 프레임의 평균을 임시 정면으로 삼는다(관찰은 정면 방향을 설정한 직후에 시작한다).
     */
    private fun trackRawYaw(matrix: FloatArray) {
        if (matrix.size < 11 || !matrix[8].isFinite() || !matrix[10].isFinite()) { rawYawDeg = null; return }
        val yaw = Math.toDegrees(kotlin.math.atan2(matrix[8].toDouble(), matrix[10].toDouble()))
        rawYawDeg = yaw
        if (yawZeroCount < YAW_ZERO_FRAMES) { yawZeroSum += yaw; yawZeroCount++ }
    }

    /** 임시 정면 대비 좌우 각도. 임시 정면을 아직 못 잡았으면 null. */
    private fun provisionalYawDeg(): Double? {
        val yaw = rawYawDeg ?: return null
        if (yawZeroCount < YAW_ZERO_FRAMES) return null
        return yaw - yawZeroSum / yawZeroCount
    }

    // ── 기준이 잡히지 않았을 때의 전방 주시 판정 ──
    private var fallbackAway = false
    private var awaySinceMs: Long? = null
    private var lastYawAway = false

    /**
     * 라이브러리가 머리 방향을 판정하지 못하는 동안(평소 자세를 익히는 첫 1분, 또는 익히기에 실패해 공통 기준으로 넘어간 경우)의
     * 전방 주시 이탈 판정. 그대로 두면 이 구간에는 고개를 돌려도 경고가 전혀 나지 않는다.
     * 관찰 시작 직후의 방향(정면 방향을 설정한 직후)을 임시 정면으로 삼아, [FALLBACK_AWAY_DEG] 넘게 [FALLBACK_AWAY_MS] 이상
     * 돌리고 있으면 이탈로, [FALLBACK_BACK_DEG] 안으로 돌아오면 복귀로 알린다.
     * 라이브러리와 같이 저속(시속 20km 미만)에서는 경고하지 않는다. 기준이 잡히면 라이브러리 판정으로 넘어간다.
     */
    private fun fallbackGaze(face: Boolean, poseValid: Boolean, sessionMs: Long, speedKmh: Double?) {
        val yaw = provisionalYawDeg()?.takeIf { face }
        // 고개를 크게 돌리면 얼굴을 놓치기도 한다. 그때는 직전 상태(돌리고 있었는지)를 이어 간다.
        val away = if (yaw != null) kotlin.math.abs(yaw) > FALLBACK_AWAY_DEG else lastYawAway
        lastYawAway = away
        if (fallbackAway) {
            val back = yaw != null && kotlin.math.abs(yaw) < FALLBACK_BACK_DEG
            if (back || (poseValid && !away)) {
                fallbackAway = false
                lookingAway = false
                awaySinceMs = null
                onDetection(Detection.LOOK_BACK)
            }
            return
        }
        if (!away || lookingAway) { awaySinceMs = null; return }
        val since = awaySinceMs ?: sessionMs.also { awaySinceMs = it }
        if (sessionMs - since < FALLBACK_AWAY_MS) return
        if (speedKmh != null && speedKmh < FALLBACK_MIN_SPEED_KMH) return
        Log.i(TAG, "t=${sessionMs}ms 임시 기준으로 전방 주시 이탈(기준이 아직 없음)")
        fallbackAway = true
        lookingAway = true
        onDetection(Detection.LOOK_AWAY)
    }

    private companion object {
        const val TAG = "DrowsyDetector"
        const val FALLBACK_AWAY_DEG = 30.0
        const val FALLBACK_BACK_DEG = 15.0
        const val FALLBACK_AWAY_MS = 3_000L
        const val FALLBACK_MIN_SPEED_KMH = 20.0
        /** 임시 정면을 잡는 프레임 수(약 1초) */
        const val YAW_ZERO_FRAMES = 15
        /** 분석 간격(약 15fps). 라이브러리는 프레임 간격 200ms 이하를 요구한다. */
        const val MIN_INTERVAL_MS = 60L
        /** 결과가 이 시간 넘게 오지 않으면 그 프레임은 버려진 것으로 본다 */
        const val STUCK_MS = 1_000L
    }
}
