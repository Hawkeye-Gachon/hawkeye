package com.personal.driverguard.views.home

import android.Manifest
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.automirrored.rounded.List
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.R
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.views.base.EventEffect
import com.personal.driverguard.views.base.appViewModel
import com.personal.driverguard.views.components.AdaptiveLayout
import com.personal.driverguard.views.components.GlassSurface
import com.personal.driverguard.views.components.GroupRow
import com.personal.driverguard.views.components.InsetGroup
import com.personal.driverguard.views.components.PermissionHandle
import com.personal.driverguard.views.components.PermissionStatus
import com.personal.driverguard.views.components.PrimaryButton
import com.personal.driverguard.views.components.SecondaryButton
import com.personal.driverguard.views.components.SkyScaffold
import com.personal.driverguard.views.components.CapsuleShape
import com.personal.driverguard.views.components.fadeBottomEdge
import com.personal.driverguard.views.components.rememberPermission
import com.personal.driverguard.views.theme.sizes
import com.personal.driverguard.views.theme.spacing

/**
 * 홈(정차 상태). 주요 동작은 하나: 주행 시작.
 * 하늘 위 왼쪽에 제목·분석 방식 설명·시작, 오른쪽에 준비 상태 묶음. 기록·설정은 오른쪽 위 유리 캡슐.
 */
@Composable
fun HomeScreen(
    onCalibrate: () -> Unit,
    onOpenTrips: () -> Unit,
    onOpenDevSettings: () -> Unit,
    viewModel: HomeViewModel = appViewModel(),
) {
    val modelStatus by viewModel.modelStatus.collectAsStateWithLifecycle()
    val camera = rememberPermission(Manifest.permission.CAMERA)
    val location = rememberPermission(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
    val modelReady = modelStatus is ModelAssetStatus.Ready

    EventEffect(viewModel.events) { event ->
        when (event) {
            HomeEvent.Calibrate -> onCalibrate()
        }
    }

    // 카메라가 없으면 주요 버튼이 곧 권한 요청이 된다. 분석은 카메라 위치 확인부터 시작한다.
    val needsCamera = camera.status != PermissionStatus.GRANTED
    val primaryLabel = when {
        !needsCamera -> stringResource(R.string.action_check_camera)
        camera.status == PermissionStatus.BLOCKED -> stringResource(R.string.action_allow_camera_in_settings)
        else -> stringResource(R.string.action_allow_camera)
    }
    val onPrimary = { if (needsCamera) camera.request() else viewModel.start(cameraGranted = true) }

    val intro: @Composable ColumnScope.() -> Unit = {
        Text(
            stringResource(R.string.app_name),
            style = MaterialTheme.typography.displayMedium,
            modifier = Modifier.semantics { heading() },
        )
        Text(stringResource(R.string.home_tagline), style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(MaterialTheme.spacing.m))
        Text(
            stringResource(R.string.mode_real_description),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
    val start: @Composable ColumnScope.() -> Unit = {
        PrimaryButton(
            primaryLabel,
            onPrimary,
            modifier = Modifier.widthIn(max = MaterialTheme.sizes.homeStartPaneMaxWidth).fillMaxWidth(),
            enabled = modelReady || needsCamera,
        )
    }
    val readinessGroup: @Composable ColumnScope.() -> Unit = {
        InsetGroup(header = stringResource(R.string.home_readiness)) {
            // 카메라가 없으면 시작 버튼이 곧 권한 요청이므로 줄에는 버튼을 두지 않는다
            PermissionRow(stringResource(R.string.permission_camera), camera, R.string.permission_camera_granted, R.string.permission_camera_denied, required = true, divider = false, showAction = false)
            PermissionRow(stringResource(R.string.permission_location), location, R.string.permission_location_granted, R.string.permission_location_denied, required = false, divider = true, showAction = true)
            GroupRow(
                title = stringResource(R.string.readiness_model),
                subtitle = when (val m = modelStatus) {
                    null -> stringResource(R.string.model_checking)
                    is ModelAssetStatus.Ready -> stringResource(R.string.readiness_model_ready)
                    ModelAssetStatus.Missing -> stringResource(R.string.model_missing)
                    is ModelAssetStatus.Failed -> m.message
                },
                leading = {
                    if (modelStatus == null) {
                        CircularProgressIndicator(Modifier.size(MaterialTheme.sizes.iconM), color = MaterialTheme.colorScheme.primary)
                    } else {
                        RowIcon(if (modelReady) Icons.Rounded.CheckCircle else Icons.Rounded.Warning, warn = !modelReady)
                    }
                },
                trailing = { Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant) },
                onClick = onOpenDevSettings,
                showDivider = true,
            )
            GroupRow(
                title = stringResource(R.string.readiness_calibration),
                subtitle = stringResource(R.string.readiness_calibration_body),
                leading = { RowIcon(Icons.Rounded.Info, warn = false) },
                showDivider = true,
            )
        }
    }
    val corner: @Composable () -> Unit = {
        // 기록·설정: 하늘 위에 뜬 유리 캡슐(조작층)
        GlassSurface(shape = CapsuleShape) {
            Row(Modifier.padding(MaterialTheme.spacing.xxs), horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xxs)) {
                SecondaryButton(stringResource(R.string.action_trips), onOpenTrips, Modifier, leading = Icons.AutoMirrored.Rounded.List)
                SecondaryButton(stringResource(R.string.action_dev_settings), onOpenDevSettings, Modifier, leading = Icons.Rounded.Settings)
            }
        }
    }

    SkyScaffold(glowIntensity = 0.55f, dim = 0f) { padding ->
        AdaptiveLayout(
            modifier = Modifier.fillMaxSize().padding(padding),
            wide = {
                // 화면 전체가 스크롤 하나다(두 열이 따로 스크롤되지 않는다).
                // 공간이 남으면 시작 버튼을 아래로 밀고, 큰 글자에서는 겹치지 않고 함께 스크롤된다.
                BoxWithConstraints(Modifier.fillMaxSize()) {
                    val scroll = rememberScrollState()
                    val viewport = maxHeight
                    Row(
                        Modifier
                            .fillMaxSize()
                            .fadeBottomEdge(enabled = scroll.canScrollForward, fraction = 0.08f)
                            .verticalScroll(scroll)
                            // 최소 높이(화면)를 먼저 걸고 그 안에서 내용 높이를 쓴다 → max(화면, 내용)
                            .heightIn(min = viewport)
                            .height(IntrinsicSize.Min)
                            .padding(MaterialTheme.spacing.l),
                        horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xl),
                    ) {
                        // 오른쪽 목록이 화면보다 길어도 시작 버튼은 첫 화면 안에 둔다(왼쪽 열 높이를 화면 높이로 제한)
                        Column(
                            Modifier.weight(1f).heightIn(max = viewport - MaterialTheme.spacing.l * 2).fillMaxHeight(),
                            verticalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Column { intro() }
                            Column(Modifier.padding(top = MaterialTheme.spacing.m)) { start() }
                        }
                        Column(
                            Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m),
                            horizontalAlignment = Alignment.End,
                        ) {
                            corner()
                            readinessGroup()
                        }
                    }
                }
            },
            compact = {
                Column(Modifier.fillMaxSize()) {
                    val scroll = rememberScrollState()
                    Column(
                        Modifier.weight(1f).fadeBottomEdge(enabled = scroll.canScrollForward).verticalScroll(scroll).padding(MaterialTheme.spacing.m),
                        verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s),
                    ) {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) { corner() }
                        intro()
                        Spacer(Modifier.height(MaterialTheme.spacing.m))
                        readinessGroup()
                    }
                    Column(Modifier.padding(MaterialTheme.spacing.m)) { start() }
                }
            },
        )
    }
}

@Composable
private fun RowIcon(icon: ImageVector, warn: Boolean) {
    Icon(
        icon,
        contentDescription = null,
        // 앰버·레드는 운전자 상태 전용 예약색이라 설정 경고에는 쓰지 않는다. 다급함은 버튼이 전한다.
        tint = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.size(MaterialTheme.sizes.iconM),
    )
}

@Composable
private fun PermissionRow(title: String, permission: PermissionHandle, grantedText: Int, deniedText: Int, required: Boolean, divider: Boolean, showAction: Boolean) {
    val granted = permission.status == PermissionStatus.GRANTED
    // 선택 권한이 없는 것은 오류가 아니므로 경고색을 쓰지 않는다
    val warn = !granted && required
    GroupRow(
        title = title,
        subtitle = stringResource(if (granted) grantedText else deniedText),
        leading = { RowIcon(if (granted) Icons.Rounded.CheckCircle else if (warn) Icons.Rounded.Warning else Icons.Rounded.Info, warn) },
        trailing = if (granted || !showAction) {
            null
        } else {
            {
                SecondaryButton(
                    text = stringResource(if (permission.status == PermissionStatus.BLOCKED) R.string.action_open_settings else R.string.action_allow),
                    onClick = permission.request,
                    modifier = Modifier,
                )
            }
        },
        showDivider = divider,
    )
}

