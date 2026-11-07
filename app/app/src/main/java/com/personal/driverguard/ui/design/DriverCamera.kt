package com.personal.driverguard.ui.design

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.personal.driverguard.platform.android.camera.CameraBinding
import com.personal.driverguard.views.components.PermissionStatus
import com.personal.driverguard.views.components.rememberPermission
import java.util.concurrent.Executors

/**
 * 카메라 프레임을 받는 곳. 지금은 프레임을 바로 닫기만 한다.
 * AI 추론 서버가 준비되면 여기서 프레임을 잘라 서버로 보낸다(분석 결과로 화면 상태를 바꾸는 연결은 아직 없다).
 */
object DriverFrames {
    @Volatile
    var sink: ((ImageProxy) -> Unit)? = null

    val analyzer = ImageAnalysis.Analyzer { image ->
        try {
            sink?.invoke(image)
        } finally {
            image.close()
        }
    }
}

/** 권한 요청은 앱 실행마다 한 번만 자동으로 띄운다(화면마다 다시 묻지 않는다). 위치(쉼터 조회·경로 기록)도 함께 묻는다. */
private object CameraPermissionPrompt {
    var asked = false
}

/**
 * 운전자 전면 카메라 영상(거울상 미리보기). 임시 연결이다.
 * 카메라 권한이 없거나, 카메라가 없거나, 연결에 실패하면 [fallback](시안의 예시 사진)을 그린다.
 * 그래서 에뮬레이터와 스크린샷 테스트에서도 화면 흐름을 그대로 확인할 수 있다.
 */
@Composable
fun DriverCamera(modifier: Modifier = Modifier, fallback: @Composable BoxScope.() -> Unit) {
    val permission = rememberPermission(Manifest.permission.CAMERA)
    var failed by remember { mutableStateOf(false) }
    val executor = remember { Executors.newSingleThreadExecutor { Thread(it, "driver-camera") } }
    DisposableEffect(executor) { onDispose { executor.shutdown() } }

    // 카메라와 위치를 한 번에 묻는다(두 요청을 따로 띄우면 하나가 무시될 수 있다). 결과는 화면 복귀 시 permission 이 다시 읽는다.
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }
    LaunchedEffect(permission.status) {
        if (permission.status == PermissionStatus.REQUESTABLE && !CameraPermissionPrompt.asked) {
            CameraPermissionPrompt.asked = true
            launcher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        }
    }

    Box(modifier) {
        if (permission.status == PermissionStatus.GRANTED && !failed) {
            CameraBinding(
                analyzer = DriverFrames.analyzer,
                analysisExecutor = executor,
                showPreview = true,
                onError = { failed = true },
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            fallback()
        }
    }
}
