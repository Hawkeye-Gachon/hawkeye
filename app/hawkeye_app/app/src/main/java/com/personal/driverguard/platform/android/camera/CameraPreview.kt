package com.personal.driverguard.platform.android.camera

import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executor

/**
 * 전면 카메라를 화면 생명주기에 바인딩한다. 분석 프레임은 항상 [analyzer] 로 보내고,
 * [showPreview] 일 때만 미리보기를 거울상으로 보여 준다(PreviewView 기본 동작).
 * 화면이 백그라운드로 가면 CameraX 가 카메라를 멈추고, 컴포저블이 사라지면 바인딩을 해제한다.
 *
 * 분석 프레임과 미리보기는 둘 다 4:3 이라 오버레이 좌표를 같은 비율로 맞출 수 있다.
 */
@Composable
fun CameraBinding(
    analyzer: ImageAnalysis.Analyzer,
    analysisExecutor: Executor,
    showPreview: Boolean,
    onError: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnError by rememberUpdatedState(onError)
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    DisposableEffect(lifecycleOwner, analyzer, showPreview) {
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        var disposed = false
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                    .build(),
            )
            .build()
            .also { it.setAnalyzer(analysisExecutor, analyzer) }
        val preview = if (showPreview) {
            Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build())
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
        } else {
            null
        }
        val useCases: List<UseCase> = listOfNotNull(preview, analysis)

        future.addListener({
            if (disposed) return@addListener
            try {
                val cameraProvider = future.get()
                if (!cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                    currentOnError("전면 카메라를 찾을 수 없습니다")
                    return@addListener
                }
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, *useCases.toTypedArray())
                provider = cameraProvider
            } catch (e: Exception) {
                currentOnError("카메라 연결 실패: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            disposed = true
            analysis.clearAnalyzer()
            provider?.unbind(*useCases.toTypedArray())
        }
    }

    if (showPreview) AndroidView(factory = { previewView }, modifier = modifier) else Box(modifier)
}
