package com.personal.driverguard.views.driving

import android.os.SystemClock
import androidx.activity.compose.BackHandler
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.personal.driverguard.views.components.SecondaryIconButton
import androidx.compose.material.icons.rounded.Face
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.R
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.DrowsinessLevel
import com.personal.driverguard.platform.android.camera.CameraBinding
import com.personal.driverguard.runtime.LiveAnalysisSession
import com.personal.driverguard.views.base.EventEffect
import com.personal.driverguard.views.base.appViewModel
import com.personal.driverguard.views.components.AdaptiveLayout
import com.personal.driverguard.views.components.CapsuleShape
import com.personal.driverguard.views.components.ConfirmDialog
import com.personal.driverguard.views.components.FaceOverlayView
import com.personal.driverguard.views.components.GlassSurface
import com.personal.driverguard.views.components.KeepScreenOn
import com.personal.driverguard.views.components.LocalHaze
import com.personal.driverguard.views.components.PrimaryButton
import com.personal.driverguard.views.components.PrivacyChip
import com.personal.driverguard.views.components.SecondaryButton
import com.personal.driverguard.views.components.SkyBackground
import com.personal.driverguard.views.components.displayGuidance
import com.personal.driverguard.views.components.displayIcon
import com.personal.driverguard.views.components.displayLabel
import com.personal.driverguard.views.components.formatClock
import com.personal.driverguard.views.components.qualityLabel
import com.personal.driverguard.views.theme.TabularNumbers
import com.personal.driverguard.views.theme.details
import com.personal.driverguard.views.theme.sizes
import com.personal.driverguard.views.theme.spacing
import com.personal.driverguard.views.theme.status
import dev.chrisbanes.haze.hazeSource
import dev.chrisbanes.haze.rememberHazeState
import kotlinx.coroutines.delay

/**
 * 주행 화면. 한눈에 읽는 화면이라 스크롤하지 않는다.
 * 하늘 필드가 화면 전체를 차지하고, 상태는 지평선 빛과 큰 글자로 말한다.
 * 조작(휴식·종료)은 하늘 위에 뜬 유리 레일이다. 경고 상태에서는 그 밖의 요소가 물러난다.
 * 카메라 영상은 기본으로 숨기고, 켜면 영상 위에는 아무것도 덮지 않은 창으로 보여 주며 상태 문구는 영상 밖에 둔다.
 */
@Composable
fun DrivingScreen(
    onFinished: (Long) -> Unit,
    onExit: () -> Unit,
    viewModel: DrivingViewModel = appViewModel(),
) {
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    val session by viewModel.session.collectAsStateWithLifecycle()
    var confirmFinish by rememberSaveable { mutableStateOf(false) }

    KeepScreenOn()
    // 백그라운드 분석은 하지 않는다. 화면을 벗어나면 중단 구간으로 기록한다.
    LifecycleEventEffect(Lifecycle.Event.ON_START) { viewModel.onVisibilityChanged(true) }
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { viewModel.onVisibilityChanged(false) }
    BackHandler { if (ui.phase == DrivingPhase.RUNNING) confirmFinish = true else viewModel.requestFinish() }
    EventEffect(viewModel.events) { event ->
        when (event) {
            is DrivingEvent.Finished -> onFinished(event.tripId)
            DrivingEvent.Exit -> onExit()
        }
    }
    val requestFinish = { if (ui.phase == DrivingPhase.RUNNING) confirmFinish = true else viewModel.requestFinish() }

    val status = MaterialTheme.status
    val running = ui.phase == DrivingPhase.RUNNING
    val glow = if (running) status.glow(ui.display) else status.slate
    val intensity = when {
        !running -> 0.35f
        ui.display == DisplayState.CAUTION -> 0.95f
        ui.display == DisplayState.REST_RECOMMENDED -> 1f
        ui.display == DisplayState.OBSERVING -> 0.55f
        else -> 0.5f
    }
    // 경고 상태에서는 상태와 휴식 외 요소가 한 단계 물러난다
    val alert = running && (ui.display == DisplayState.CAUTION || ui.display == DisplayState.REST_RECOMMENDED)
    val recede by animateFloatAsState(if (alert) RECEDED_ALPHA else 1f, if (MaterialTheme.details.reduceMotion) snap() else tween(RECEDE_MS), label = "recede")

    // 카메라는 항상 바인딩해 분석 프레임을 받고, 미리보기는 켰을 때만 그린다
    val current = session
    val bindCamera = running && current != null && ui.cameraError == null

    val haze = rememberHazeState()
    CompositionLocalProvider(LocalHaze provides haze) {
        Box(Modifier.fillMaxSize()) {
            SkyBackground(glow, intensity, Modifier.hazeSource(haze))
            if (bindCamera && !ui.showCamera) {
                CameraBinding(current!!.analyzer, current.executor, showPreview = false, onError = viewModel::onCameraError, modifier = Modifier.size(1.dp))
            }
            val window: @Composable (Modifier, Boolean, Boolean) -> Unit = { modifier, large, wide ->
                if (bindCamera && ui.showCamera) {
                    // 상태 문구는 영상 아래쪽 안에 둔다(얼굴이 있는 위쪽은 비워 둔다)
                    DriverWindow(current!!, ui, viewModel, glow, modifier)
                } else {
                    Box(modifier) { StatusBlock(ui, viewModel, large, glow, Modifier.align(if (large) Alignment.CenterStart else Alignment.TopStart), voiceNote = !wide) }
                }
            }
            AdaptiveLayout(
                modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(MaterialTheme.spacing.m),
                wide = {
                    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                        TopCapsule(ui, viewModel, compact = false, modifier = Modifier.alpha(recede))
                        Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                            window(Modifier.weight(1f).fillMaxHeight(), true, true)
                            // 조작 레일은 내용 크기만큼, 엄지가 닿는 아래쪽에 붙는다
                            Box(Modifier.width(MaterialTheme.sizes.drivingRailWidth).fillMaxHeight(), contentAlignment = Alignment.BottomCenter) {
                                ControlRail(ui, viewModel::toggleRest, requestFinish, Modifier.fillMaxWidth())
                            }
                        }
                    }
                },
                compact = {
                    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                        TopCapsule(ui, viewModel, compact = true, modifier = Modifier.fillMaxWidth().alpha(recede))
                        window(Modifier.weight(1f).fillMaxWidth(), false, false)
                        ControlRail(ui, viewModel::toggleRest, requestFinish, Modifier.fillMaxWidth())
                    }
                },
            )
        }
    }

    if (confirmFinish) {
        ConfirmDialog(
            title = stringResource(R.string.finish_confirm_title),
            message = stringResource(R.string.finish_confirm_message),
            confirmLabel = stringResource(R.string.action_finish_driving),
            dismissLabel = stringResource(R.string.action_keep_driving),
            onConfirm = {
                confirmFinish = false
                viewModel.requestFinish()
            },
            onDismiss = { confirmFinish = false },
        )
    }
}

/**
 * 주행 시간, 운전자 화면 보기.
 * 좁은 화면에서는 운전자 화면 버튼을 아이콘으로 줄이고, '경고음만' 안내는 상태 문구 아래로 옮긴다.
 */
@Composable
private fun TopCapsule(ui: DrivingUiState, viewModel: DrivingViewModel, compact: Boolean, modifier: Modifier = Modifier) {
    GlassSurface(modifier, shape = CapsuleShape) {
        Row(
            Modifier.padding(start = MaterialTheme.spacing.m, end = MaterialTheme.spacing.xxs, top = MaterialTheme.spacing.xxs, bottom = MaterialTheme.spacing.xxs),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s),
        ) {
            // 남는 자리는 주행 시간이 채운다(아이콘 버튼은 오른쪽 끝)
            ElapsedTime(ui.startedAtMs, if (compact) Modifier.weight(1f) else Modifier)
            if (!compact && !viewModel.voiceAvailable) {
                // TTS 를 쓸 수 없으면 경고음만 난다는 사실을 짧게 알린다
                Text(stringResource(R.string.driving_voice_off_short), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1)
            }
            if (ui.phase == DrivingPhase.RUNNING) {
                val label = stringResource(if (ui.showCamera) R.string.action_hide_camera else R.string.action_show_camera)
                if (compact) {
                    SecondaryIconButton(Icons.Rounded.Face, label, viewModel::toggleCamera, selected = ui.showCamera)
                } else {
                    SecondaryButton(label, viewModel::toggleCamera, Modifier)
                }
            }
        }
    }
}

@Composable
private fun StatusBlock(ui: DrivingUiState, viewModel: DrivingViewModel, large: Boolean, glow: Color, modifier: Modifier = Modifier, voiceNote: Boolean = false, onVideo: Boolean = false) {
    Column(
        // 주행 화면은 한눈에 읽는 화면이라 스크롤하지 않는다
        modifier.widthIn(max = MaterialTheme.sizes.contentMaxWidth),
        verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m),
    ) {
        when (ui.phase) {
            DrivingPhase.STARTING, DrivingPhase.FINISHING -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                CircularProgressIndicator(color = MaterialTheme.colorScheme.onBackground)
                Text(stringResource(if (ui.phase == DrivingPhase.STARTING) R.string.driving_starting else R.string.driving_finishing), style = MaterialTheme.typography.headlineLarge)
            }
            DrivingPhase.FAILED -> {
                StatusLine(Icons.Rounded.Warning, MaterialTheme.status.slate, stringResource(R.string.driving_failed), ui.startError, large, onVideo)
                PrimaryButton(stringResource(R.string.action_back_home), viewModel::requestFinish, Modifier.widthIn(max = MaterialTheme.sizes.homeStartPaneMaxWidth))
            }
            DrivingPhase.RUNNING -> {
                val title = stringResource(displayLabel(ui.display, ui.cautionCause))
                // 분석 불가가 쌓이는 중이면 이유를 함께 보여 준다(정상으로 보이지 않게)
                val reason = ui.unavailableReason?.takeIf { ui.display == DisplayState.UNAVAILABLE || ui.display == DisplayState.OBSERVING || ui.display == DisplayState.PREPARING }
                val detail = listOfNotNull(
                    stringResource(displayGuidance(ui.display, ui.cautionCause)),
                    reason?.let { stringResource(qualityLabel(it)) },
                ).joinToString("\n")
                StatusLine(if (ui.display == DisplayState.PAUSED) null else displayIcon(ui.display), glow, title, detail, large, onVideo)
                if (ui.drowsiness == DrowsinessLevel.SUSPECTED_WITH_SUPPORT) {
                    Text(stringResource(R.string.driving_support), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                // 좁은 화면은 상단 캡슐에 '경고음만'을 넣을 자리가 없어 여기에 적는다
                if (voiceNote && !viewModel.voiceAvailable) {
                    Text(stringResource(R.string.driving_voice_off), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (ui.display == DisplayState.REST_RECOMMENDED && viewModel.restAreaGuideAvailable) {
                    SecondaryButton(stringResource(R.string.action_search_rest_area), viewModel::openRestAreaSearch, Modifier)
                }
                ui.cameraError?.let { Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error) }
            }
        }
    }
}

/** 상태 아이콘(지평선 빛과 같은 색) + 큰 제목 + 안내. 바뀔 때 TalkBack 이 읽도록 라이브 영역으로 묶는다. */
@Composable
private fun StatusLine(icon: ImageVector?, tint: Color, title: String, detail: String?, large: Boolean, onVideo: Boolean = false) {
    // 영상 위 글자는 밝은 배경에서도 읽히도록 옅은 그림자를 준다(블러 막 대신)
    val shadow = if (onVideo) Shadow(Color.Black.copy(alpha = TEXT_SHADOW_ALPHA), blurRadius = TEXT_SHADOW_BLUR) else null
    Column(Modifier.semantics(mergeDescendants = true) { liveRegion = LiveRegionMode.Polite }, verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xs)) {
        if (large) icon?.let { Icon(it, contentDescription = null, tint = tint, modifier = Modifier.size(MaterialTheme.sizes.iconL)) }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s)) {
            if (!large) icon?.let { Icon(it, contentDescription = null, tint = tint, modifier = Modifier.size(MaterialTheme.sizes.iconM)) }
            Text(title, style = (if (large) MaterialTheme.typography.displaySmall else MaterialTheme.typography.headlineLarge).copy(shadow = shadow))
        }
        detail?.let {
            Text(
                it,
                style = (if (large) MaterialTheme.typography.headlineSmall else MaterialTheme.typography.titleMedium).copy(shadow = shadow),
                // 영상 위에서는 대비를 위해 보조색 대신 본문색을 쓴다
                color = if (onVideo) MaterialTheme.colorScheme.onBackground else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/** 조작 레일(유리): 휴식이 강조 동작, 종료는 보조 동작. 두 버튼은 같은 높이(56dp)다. */
@Composable
private fun ControlRail(ui: DrivingUiState, onToggleRest: () -> Unit, onFinish: () -> Unit, modifier: Modifier = Modifier) {
    GlassSurface(modifier, shape = MaterialTheme.shapes.extraLarge) {
        // 유리 컨테이너는 안쪽 4dp·간격 4dp. 바깥 반경 32 = 버튼 반경 28 + 4 (동심 모서리)
        Column(Modifier.fillMaxWidth().padding(MaterialTheme.spacing.xxs), verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xxs, Alignment.Bottom)) {
            if (ui.phase == DrivingPhase.RUNNING) {
                PrimaryButton(stringResource(if (ui.isResting) R.string.action_end_rest else R.string.action_start_rest), onToggleRest)
            }
            if (ui.phase == DrivingPhase.RUNNING || ui.phase == DrivingPhase.STARTING) {
                SecondaryButton(stringResource(R.string.action_finish_driving), onFinish)
            }
        }
    }
}

/**
 * 운전석 창(선택 표시): 운전자를 비추는 카메라 영상 안에 상태 문구를 함께 보여 준다.
 * 얼굴이 잘 보이도록 블러는 쓰지 않고, 글자가 놓이는 아래 가장자리에만 옅은 그라데이션을 깐다.
 * 창 테두리가 상태색을 따른다. 영상과 얼굴 이미지는 저장하거나 전송하지 않는다.
 */
@Composable
private fun DriverWindow(session: LiveAnalysisSession, ui: DrivingUiState, viewModel: DrivingViewModel, glow: Color, modifier: Modifier) {
    val overlay by session.overlay.collectAsStateWithLifecycle()
    val edge by animateColorAsState(glow, if (MaterialTheme.details.reduceMotion) snap() else tween(GLOW_MS), label = "windowEdge")
    val shape = MaterialTheme.shapes.extraLarge
    val scrim = MaterialTheme.colorScheme.background
    BoxWithConstraints(modifier.clip(shape).background(MaterialTheme.colorScheme.surfaceContainerLowest).border(WINDOW_EDGE, edge.copy(alpha = 0.75f), shape)) {
        CameraBinding(session.analyzer, session.executor, showPreview = true, onError = viewModel::onCameraError, modifier = Modifier.fillMaxSize())
        FaceOverlayView(overlay, edge, Modifier.fillMaxSize())
        PrivacyChip(Modifier.align(Alignment.TopStart).padding(MaterialTheme.spacing.s), compact = maxWidth < PRIVACY_TEXT_MIN_WIDTH)
        // 그라데이션은 글자 높이만큼만 깐다(창이 낮아도 글자 뒤는 항상 덮이고, 얼굴 쪽은 비워 둔다)
        Box(
            Modifier
                .align(Alignment.BottomStart)
                .fillMaxWidth()
                .background(Brush.verticalGradient(0f to Color.Transparent, SCRIM_SOLID_FROM to scrim.copy(alpha = SCRIM_ALPHA), 1f to scrim.copy(alpha = SCRIM_ALPHA)))
                .padding(start = MaterialTheme.spacing.m, end = MaterialTheme.spacing.m, top = MaterialTheme.spacing.xl, bottom = MaterialTheme.spacing.m),
        ) {
            StatusBlock(ui, viewModel, large = false, glow = glow, onVideo = true)
        }
    }
}

@Composable
private fun ElapsedTime(startedAtMs: Long?, modifier: Modifier = Modifier) {
    val now by produceState(SystemClock.elapsedRealtime(), startedAtMs) {
        while (true) {
            value = SystemClock.elapsedRealtime()
            delay(1_000)
        }
    }
    val text = startedAtMs?.let { formatClock(now - it) } ?: "--:--"
    // 화면에는 '주행 05:09', 스크린 리더도 같은 문장으로 읽는다
    val spoken = stringResource(R.string.driving_elapsed, text)
    Text(spoken, style = MaterialTheme.typography.titleMedium.merge(TabularNumbers), maxLines = 1, softWrap = false, modifier = modifier.semantics { contentDescription = spoken })
}

private const val GLOW_MS = 700
private val WINDOW_EDGE = 2.dp
private const val RECEDED_ALPHA = 0.55f
private const val RECEDE_MS = 300

/** 영상 안 상태 문구 뒤 그라데이션: 글자 영역 위쪽에서 시작해 글자 뒤에서는 이 농도로, 영상이 비쳐 보이게 */
private const val SCRIM_SOLID_FROM = 0.4f
private const val SCRIM_ALPHA = 0.7f
private const val TEXT_SHADOW_ALPHA = 0.55f
private const val TEXT_SHADOW_BLUR = 8f

/** 카메라 창이 이보다 좁으면 개인정보 표시를 아이콘으로 줄인다 */
private val PRIVACY_TEXT_MIN_WIDTH = 220.dp
