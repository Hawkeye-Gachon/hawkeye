package com.personal.driverguard.views.devsettings

import android.Manifest
import androidx.annotation.StringRes
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.R
import com.personal.driverguard.analysis.FaceModelAsset
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.domain.engine.EngineSnapshot
import com.personal.driverguard.platform.android.camera.CameraBinding
import com.personal.driverguard.views.base.EventEffect
import com.personal.driverguard.views.base.appViewModel
import com.personal.driverguard.views.components.AdaptiveLayout
import com.personal.driverguard.views.components.CapsuleShape
import com.personal.driverguard.views.components.plus
import com.personal.driverguard.views.components.FaceOverlayView
import com.personal.driverguard.views.components.GlassTopBar
import com.personal.driverguard.views.components.GroupRow
import com.personal.driverguard.views.components.InsetGroup
import com.personal.driverguard.views.components.LoadingState
import com.personal.driverguard.views.components.MessageState
import com.personal.driverguard.views.components.PermissionStatus
import com.personal.driverguard.views.components.PrimaryButton
import com.personal.driverguard.views.components.SecondaryButton
import com.personal.driverguard.views.components.SegmentedControl
import com.personal.driverguard.views.components.SkyScaffold
import com.personal.driverguard.views.components.scrollEdge
import com.personal.driverguard.views.components.GlassSurface
import com.personal.driverguard.views.components.formatMs
import com.personal.driverguard.views.components.qualityLabel
import com.personal.driverguard.views.components.rememberPermission
import com.personal.driverguard.views.theme.TabularNumbers
import com.personal.driverguard.views.theme.glass
import com.personal.driverguard.views.theme.sizes
import com.personal.driverguard.views.theme.spacing
import com.personal.driverguard.views.theme.status
import kotlinx.coroutines.launch
import java.util.Locale

private enum class DevTab(@param:StringRes val title: Int) {
    LIVE(R.string.dev_tab_live),
    THRESHOLDS(R.string.dev_tab_thresholds),
    INFO(R.string.dev_tab_info),
}

/** 개발 화면(정차·개발 환경 전용). 홈에서만 들어온다. */
@Composable
fun DevSettingsScreen(onBack: () -> Unit, viewModel: DevViewModel = appViewModel()) {
    var tabIndex by rememberSaveable { mutableIntStateOf(0) }
    val tab = DevTab.entries[tabIndex]
    val snackbar = remember { SnackbarHostState() }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    EventEffect(viewModel.events) { event ->
        val message = when (event) {
            DevEvent.Saved -> context.getString(R.string.dev_saved)
            DevEvent.Reset -> context.getString(R.string.dev_reset_done)
            is DevEvent.LogStopped -> context.getString(R.string.dev_log_saved, event.path)
            is DevEvent.LogsDeleted -> context.getString(R.string.dev_logs_deleted, event.count)
        }
        scope.launch { snackbar.showSnackbar(message) }
    }

    Box(Modifier.fillMaxSize()) {
        SkyScaffold(
            topBar = {
                GlassTopBar(
                    title = stringResource(R.string.action_dev_settings),
                    onBack = onBack,
                    below = {
                        SegmentedControl(
                            options = DevTab.entries,
                            selected = tab,
                            label = { stringResource(it.title) },
                            onSelect = { tabIndex = it.ordinal },
                            modifier = Modifier.widthIn(max = MaterialTheme.sizes.contentMaxWidth).fillMaxWidth().align(Alignment.Center),
                        )
                    },
                )
            },
            bottomBar = {
                if (tab == DevTab.THRESHOLDS) {
                    // 떠 있는 유리 캡슐 바. 바 뒤로 스크롤되는 목록은 아래쪽 스크롤 가장자리 효과로 흐려진다.
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .scrollEdge(fromTop = false)
                            .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Bottom + WindowInsetsSides.Horizontal))
                            .padding(MaterialTheme.spacing.m),
                        contentAlignment = Alignment.Center,
                    ) {
                        GlassSurface(Modifier.widthIn(max = MaterialTheme.sizes.homeStartPaneMaxWidth).fillMaxWidth(), shape = CapsuleShape) {
                            Row(
                                Modifier.fillMaxWidth().padding(MaterialTheme.spacing.xxs),
                                horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xxs),
                            ) {
                                SecondaryButton(stringResource(R.string.action_reset_defaults), viewModel::resetToDefaults, Modifier.weight(1f))
                                PrimaryButton(stringResource(R.string.action_save), viewModel::save, Modifier.weight(1f))
                            }
                        }
                    }
                }
            },
        ) { padding ->
            // 목록 탭은 바 여백을 스크롤 안쪽에 둔다
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                when (tab) {
                    DevTab.LIVE -> Box(Modifier.fillMaxSize().padding(padding)) { LiveTab(viewModel) }
                    DevTab.THRESHOLDS -> ThresholdsTab(viewModel, padding)
                    DevTab.INFO -> InfoTab(viewModel, padding)
                }
            }
        }
        SnackbarHost(snackbar, Modifier.align(Alignment.BottomCenter).padding(bottom = MaterialTheme.sizes.control + MaterialTheme.spacing.xl))
    }
}

@Composable
private fun TabColumn(modifier: Modifier = Modifier, barPadding: PaddingValues = PaddingValues(), content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(barPadding.plus(MaterialTheme.spacing.m))
            .wrapContentWidth()
            .widthIn(max = MaterialTheme.sizes.contentMaxWidth),
        verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.l),
        content = content,
    )
}

/* ------------------------------------------------------------------ 실시간 분석 */

@Composable
private fun LiveTab(viewModel: DevViewModel) {
    val camera = rememberPermission(Manifest.permission.CAMERA)
    val live by viewModel.live.collectAsStateWithLifecycle()
    val session by viewModel.session.collectAsStateWithLifecycle()
    var showIndices by rememberSaveable { mutableStateOf(false) }

    if (camera.status != PermissionStatus.GRANTED) {
        MessageState(
            icon = Icons.Rounded.Info,
            title = stringResource(R.string.dev_live_need_camera),
            body = stringResource(R.string.dev_live_note),
            actionLabel = stringResource(R.string.action_allow_camera),
            onAction = camera.request,
        )
        return
    }
    LaunchedEffect(Unit) { viewModel.startLive() }
    live.error?.let { error ->
        MessageState(icon = Icons.Rounded.Info, title = stringResource(R.string.dev_live_failed), body = error)
        return
    }
    val current = session
    if (current == null) {
        LoadingState(stringResource(R.string.driving_starting))
        return
    }
    val overlay by current.overlay.collectAsStateWithLifecycle()

    val cameraPane: @Composable (Modifier) -> Unit = { modifier ->
        Box(modifier.clip(MaterialTheme.shapes.large).background(MaterialTheme.colorScheme.surfaceContainerLowest)) {
            CameraBinding(current.analyzer, current.executor, showPreview = true, onError = viewModel::onCameraError, modifier = Modifier.fillMaxSize())
            FaceOverlayView(overlay, MaterialTheme.status.clear, Modifier.fillMaxSize(), debug = true, showIndices = showIndices)
        }
    }
    val controls: @Composable ColumnScope.() -> Unit = {
        Text(stringResource(R.string.dev_live_note), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s)) {
            Switch(checked = showIndices, onCheckedChange = { showIndices = it })
            Text(stringResource(R.string.dev_show_indices), style = MaterialTheme.typography.bodyMedium)
        }
        // 기준값 측정은 한 줄 전체, 로그 시작·삭제는 나란히(글자가 줄바꿈되지 않게)
        SecondaryButton(stringResource(R.string.dev_recalibrate), viewModel::recalibrate, Modifier.fillMaxWidth(), enabled = !live.calibrating)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s)) {
            SecondaryButton(stringResource(if (live.logging) R.string.dev_log_stop else R.string.dev_log_start), viewModel::toggleLog, Modifier.weight(1f))
            SecondaryButton(stringResource(R.string.dev_log_delete, live.logFiles), viewModel::deleteLogs, Modifier.weight(1f), enabled = live.logFiles > 0)
        }
        live.calibrationMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        if (!live.hasBaseline) {
            Text(stringResource(R.string.dev_no_baseline), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        MetricsGroup(live.snapshot)
    }

    AdaptiveLayout(
        modifier = Modifier.fillMaxSize(),
        wide = {
            Row(Modifier.fillMaxSize().padding(MaterialTheme.spacing.m), horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                cameraPane(Modifier.weight(1f).fillMaxHeight())
                Column(
                    Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m),
                    content = controls,
                )
            }
        },
        compact = {
            TabColumn {
                cameraPane(Modifier.fillMaxWidth().aspectRatio(3f / 4f))
                controls()
            }
        },
    )
}

@Composable
private fun MetricsGroup(s: EngineSnapshot?) {
    fun f(v: Double?, digits: Int = 3) = v?.let { String.format(Locale.US, "%.${digits}f", it) } ?: "-"
    val rows: List<Pair<String, String>> = if (s == null) {
        listOf(stringResource(R.string.dev_metric_waiting) to "")
    } else {
        val t = s.thresholds
        val r = s.closedRatio
        listOf(
            stringResource(R.string.dev_metric_quality) to listOf(
                "얼굴 " + stringResource(qualityLabel(s.faceQuality)),
                "눈 " + stringResource(qualityLabel(s.eyeQuality)),
                "입 " + stringResource(qualityLabel(s.mouthQuality)),
                "자세 " + stringResource(qualityLabel(s.poseQuality)),
            ).joinToString("\n"),
            "EAR 왼/오른" to "${f(s.leftEar)} / ${f(s.rightEar)}",
            "EAR 필터" to "${f(s.filteredLeftEar)} / ${f(s.filteredRightEar)}",
            "닫힘·열림 기준" to (t?.let { "${f(it.leftClose)}·${f(it.leftOpen)} / ${f(it.rightClose)}·${f(it.rightOpen)}" + if (it.fromClosedCalibration) "" else " *" } ?: "-"),
            "눈 상태 · 감김 지속" to "${s.eyeState} · ${s.eyeClosedMs} ms",
            "눈 감김 비율" to "${f(r.ratio)} (${r.status})",
            "유효 관측률 · 사용한 창" to "${f(r.coverage, 2)} · ${r.windowUsedMs / 1000}s",
            "입 벌림 비율 · 기준" to "${f(s.mouthRatio)} · ${f(s.mouthOpenThreshold)}",
            "입 벌림 지속" to "${s.mouthOpenMs} ms",
            "yaw · pitch · roll (기준 대비)" to "${f(s.yawDeg, 1)} · ${f(s.pitchDeg, 1)} · ${f(s.rollDeg, 1)}",
            "고개 이탈 지속" to "${s.headAwayMs} ms",
            "최근 반복(눈 감김 · 입 · 숙임)" to "${s.closureCount} · ${s.yawnCount} · ${s.nodCount}",
            "분석 불가 지속" to "${s.unavailableMs} ms",
            "졸음 판정 규칙" to s.drowsinessRules.joinToString().ifEmpty { "-" },
        )
    }
    InsetGroup(header = stringResource(R.string.dev_metrics), footer = stringResource(R.string.dev_metrics_footer)) {
        rows.forEachIndexed { i, (label, value) ->
            GroupRow(
                title = label,
                trailing = { Text(value, style = MaterialTheme.typography.bodyMedium.merge(TabularNumbers), textAlign = TextAlign.End) },
                showDivider = i > 0,
            )
        }
    }
}

/* ------------------------------------------------------------------ 임계값 */

@Composable
private fun ThresholdsTab(viewModel: DevViewModel, barPadding: PaddingValues) {
    val draft by viewModel.draft.collectAsStateWithLifecycle()
    val invalid by viewModel.invalidFields.collectAsStateWithLifecycle()
    val formErrors by viewModel.formErrors.collectAsStateWithLifecycle()
    TabColumn(barPadding = barPadding) {
        Text(stringResource(R.string.dev_thresholds_note), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        formErrors.forEach { Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error) }
        configGroups.forEach { group ->
            InsetGroup(header = stringResource(group.title)) {
                group.fields.forEachIndexed { index, field ->
                    FieldRow(field, draft[field.key].orEmpty(), field.key in invalid, index > 0, viewModel::onFieldChange)
                }
            }
        }
    }
}

/** 이름 ― 값·단위 한 줄. 값은 오른쪽 정렬 숫자 칸이다. */
@Composable
private fun FieldRow(field: ConfigField, value: String, isError: Boolean, showDivider: Boolean, onChange: (String, String) -> Unit) {
    GroupRow(
        title = field.label,
        subtitle = if (isError) stringResource(R.string.field_error_number) else field.key,
        trailing = {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xs)) {
                BasicTextField(
                    value = value,
                    onValueChange = { onChange(field.key, it) },
                    singleLine = true,
                    textStyle = MaterialTheme.typography.titleMedium.merge(TabularNumbers).copy(
                        color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
                        textAlign = TextAlign.End,
                    ),
                    cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal, imeAction = ImeAction.Next),
                    modifier = Modifier
                        .width(FIELD_WIDTH)
                        .heightIn(min = MaterialTheme.sizes.control)
                        .background(MaterialTheme.glass.fill, MaterialTheme.shapes.medium)
                        .padding(horizontal = MaterialTheme.spacing.s, vertical = MaterialTheme.spacing.s)
                        .semantics { contentDescription = field.label },
                )
                Text(field.unit, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        showDivider = showDivider,
    )
}

private val FIELD_WIDTH = 96.dp

/* ------------------------------------------------------------------ 모델·처리량 */

@Composable
private fun InfoTab(viewModel: DevViewModel, barPadding: PaddingValues) {
    val model by viewModel.modelStatus.collectAsStateWithLifecycle()
    val stats by viewModel.stats.collectAsStateWithLifecycle()
    val config by viewModel.config.collectAsStateWithLifecycle()
    TabColumn(barPadding = barPadding) {
        InsetGroup(header = stringResource(R.string.dev_model), footer = stringResource(R.string.dev_model_footer)) {
            val modelText = when (val m = model) {
                null -> stringResource(R.string.model_checking)
                is ModelAssetStatus.Ready -> "${m.fileName}\n${m.sizeBytes / 1024} KB\nSHA-256 ${m.sha256}" +
                    if (m.sha256 == FaceModelAsset.EXPECTED_SHA256) "\n" + stringResource(R.string.dev_model_hash_ok) else "\n" + stringResource(R.string.dev_model_hash_diff)
                ModelAssetStatus.Missing -> stringResource(R.string.model_missing)
                is ModelAssetStatus.Failed -> m.message
            }
            GroupRow(title = "Face Landmarker", subtitle = modelText, showDivider = false)
            GroupRow(title = stringResource(R.string.dev_config_version), subtitle = config.versionTag(), showDivider = true)
        }
        val rows = listOf(
            stringResource(R.string.dev_camera_frames) to stats.cameraFrames.toString(),
            stringResource(R.string.dev_throttled) to stats.throttled.toString(),
            stringResource(R.string.dev_dropped_busy) to stats.droppedBusy.toString(),
            stringResource(R.string.dev_analyzed) to stats.analyzed.toString(),
            stringResource(R.string.dev_throughput) to (stats.throughputFps?.let { String.format(Locale.US, "%.1f fps", it) } ?: "-"),
            stringResource(R.string.dev_inference_ms) to "${formatMs(stats.lastInferenceMs)} / ${formatMs(stats.avgInferenceMs)}",
            stringResource(R.string.dev_post_ms) to formatMs(stats.lastPostMs),
            stringResource(R.string.dev_errors) to stats.errors.toString(),
        )
        InsetGroup(header = stringResource(R.string.dev_performance), footer = listOfNotNull(stringResource(R.string.dev_performance_note), stats.lastError).joinToString("\n")) {
            rows.forEachIndexed { index, (label, value) ->
                GroupRow(
                    title = label,
                    trailing = { Text(value, style = MaterialTheme.typography.titleMedium.merge(TabularNumbers)) },
                    showDivider = index > 0,
                )
            }
        }
    }
}
