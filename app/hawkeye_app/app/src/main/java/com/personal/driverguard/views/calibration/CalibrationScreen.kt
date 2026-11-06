package com.personal.driverguard.views.calibration

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.R
import com.personal.driverguard.platform.android.camera.CameraBinding
import com.personal.driverguard.views.base.EventEffect
import com.personal.driverguard.views.base.appViewModel
import com.personal.driverguard.views.components.AdaptiveLayout
import com.personal.driverguard.views.components.FaceOverlayView
import com.personal.driverguard.views.components.GlassTopBar
import com.personal.driverguard.views.components.LoadingState
import com.personal.driverguard.views.components.MessageState
import com.personal.driverguard.views.components.PrimaryButton
import com.personal.driverguard.views.components.PrivacyChip
import com.personal.driverguard.views.components.SecondaryButton
import com.personal.driverguard.views.components.SkyScaffold
import com.personal.driverguard.views.components.calibrationFailureLabel
import com.personal.driverguard.views.components.qualityLabel
import com.personal.driverguard.views.theme.sizes
import com.personal.driverguard.views.theme.spacing
import com.personal.driverguard.views.theme.glass
import com.personal.driverguard.views.theme.status
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Warning

/**
 * 주행 전 정차 상태의 카메라 위치·기준 자세 확인.
 * 운전자를 비추는 큰 창에 얼굴 테두리를 그리고, 옆(세로 화면은 아래)에 안내·진행·결과를 둔다.
 */
@Composable
fun CalibrationScreen(
    onBack: () -> Unit,
    onStartDriving: () -> Unit,
    viewModel: CalibrationViewModel = appViewModel(),
) {
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    val session by viewModel.session.collectAsStateWithLifecycle()

    EventEffect(viewModel.events) { event ->
        when (event) {
            CalibrationEvent.StartDriving -> onStartDriving()
        }
    }

    SkyScaffold(topBar = { GlassTopBar(title = stringResource(R.string.title_calibration), onBack = onBack) }) { padding ->
        Box(Modifier.fillMaxSize().padding(padding)) {
            if (ui.step == CalibrationStep.ERROR) {
                MessageState(Icons.Rounded.Warning, stringResource(R.string.calib_error), ui.error, actionLabel = stringResource(R.string.action_back), onAction = onBack)
                return@Box
            }
            val current = session
            if (current == null) {
                LoadingState(stringResource(R.string.calib_preparing))
                return@Box
            }
            val overlay by current.overlay.collectAsStateWithLifecycle()
            val frameColor = when (ui.step) {
                CalibrationStep.DONE -> MaterialTheme.status.clear
                CalibrationStep.FAILED -> MaterialTheme.status.dawn
                else -> MaterialTheme.colorScheme.onBackground
            }
            val window: @Composable (Modifier) -> Unit = { modifier ->
                Box(modifier.clip(MaterialTheme.shapes.extraLarge).background(MaterialTheme.colorScheme.surfaceContainerLowest)) {
                    CameraBinding(current.analyzer, current.executor, showPreview = true, onError = viewModel::onCameraError, modifier = Modifier.fillMaxSize())
                    FaceOverlayView(overlay, frameColor, Modifier.fillMaxSize())
                    PrivacyChip(Modifier.align(Alignment.TopStart).padding(MaterialTheme.spacing.s))
                }
            }
            AdaptiveLayout(
                modifier = Modifier.fillMaxSize().padding(MaterialTheme.spacing.m),
                wide = {
                    Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.l)) {
                        window(Modifier.weight(1f).fillMaxHeight())
                        // 안내만 스크롤한다(카메라는 고정)
                        Column(
                            Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
                            verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m),
                        ) { GuidePanel(ui, viewModel) }
                    }
                },
                compact = {
                    // 카메라는 고정하고 안내만 스크롤한다
                    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                        window(Modifier.fillMaxWidth().weight(1f))
                        Column(
                            Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState()),
                            verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m),
                        ) { GuidePanel(ui, viewModel) }
                    }
                },
            )
        }
    }
}

@Composable
private fun ColumnScope.GuidePanel(ui: CalibrationUiState, viewModel: CalibrationViewModel) {
    Text(stringResource(R.string.calib_instruction), style = MaterialTheme.typography.headlineSmall)
    Text(stringResource(R.string.calib_stationary), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    ui.cameraError?.let { Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error) }

    Column(
        Modifier.semantics(mergeDescendants = true) { liveRegion = LiveRegionMode.Polite },
        verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xs),
    ) {
        when (ui.step) {
            CalibrationStep.PREPARING, CalibrationStep.ERROR -> Unit
            CalibrationStep.COLLECTING -> {
                LinearProgressIndicator(progress = { ui.progress.toFloat() }, modifier = Modifier.fillMaxWidth(), trackColor = MaterialTheme.glass.fill, gapSize = 0.dp, drawStopIndicator = {})
                Text(stringResource(R.string.calib_progress, (ui.progress * 100).toInt()), style = MaterialTheme.typography.titleMedium)
                ui.issue?.let { Text(stringResource(R.string.calib_issue, stringResource(qualityLabel(it))), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
            CalibrationStep.FAILED -> {
                Text(stringResource(R.string.calib_failed), style = MaterialTheme.typography.titleLarge)
                ui.failure?.let { Text(stringResource(calibrationFailureLabel(it)), style = MaterialTheme.typography.bodyMedium) }
                ui.issue?.let { Text(stringResource(qualityLabel(it)), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
            CalibrationStep.CLOSED_EYES -> {
                LinearProgressIndicator(progress = { ui.progress.toFloat() }, modifier = Modifier.fillMaxWidth(), trackColor = MaterialTheme.glass.fill, gapSize = 0.dp, drawStopIndicator = {})
                Text(stringResource(R.string.calib_closed_eye_running, (ui.progress * 100).toInt()), style = MaterialTheme.typography.titleMedium)
            }
            CalibrationStep.DONE -> {
                Text(stringResource(R.string.calib_done), style = MaterialTheme.typography.titleLarge)
                Text(stringResource(R.string.calib_done_body), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                ui.baseline?.let { Text(it.summary(), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                when {
                    ui.closedEyeFailure != null -> Text(
                        stringResource(R.string.calib_closed_eye_failed, stringResource(calibrationFailureLabel(ui.closedEyeFailure))),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    ui.baseline?.hasClosedEyeCalibration == true -> Text(stringResource(R.string.calib_closed_eye_done), style = MaterialTheme.typography.bodyMedium)
                }
            }
        }
    }

    when (ui.step) {
        CalibrationStep.FAILED -> PrimaryButton(stringResource(R.string.calib_retry), viewModel::retry)
        CalibrationStep.DONE -> {
            PrimaryButton(stringResource(R.string.action_start_driving), viewModel::confirm)
            if (ui.baseline?.hasClosedEyeCalibration != true) {
                Text(stringResource(R.string.calib_closed_eye_body), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                SecondaryButton(stringResource(R.string.calib_closed_eye), viewModel::startClosedEyeCalibration, Modifier.fillMaxWidth())
            }
            SecondaryButton(stringResource(R.string.calib_retry), viewModel::retry, Modifier.fillMaxWidth())
        }
        else -> Unit
    }
}
