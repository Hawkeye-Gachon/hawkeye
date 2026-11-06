package com.personal.driverguard.views.tripdetail

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.navigation.toRoute
import com.personal.driverguard.R
import com.personal.driverguard.domain.report.summarizeEvents
import com.personal.driverguard.domain.repository.ITripRepository
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.DriverEventVo
import com.personal.driverguard.entity.TripDetailVo
import com.personal.driverguard.views.base.appViewModel
import com.personal.driverguard.views.components.AdaptiveLayout
import com.personal.driverguard.views.components.DemoNotice
import com.personal.driverguard.views.components.GlassTopBar
import com.personal.driverguard.views.components.GroupRow
import com.personal.driverguard.views.components.InsetGroup
import com.personal.driverguard.views.components.KeyValue
import com.personal.driverguard.views.components.LoadingState
import com.personal.driverguard.views.components.MessageState
import com.personal.driverguard.views.components.ModeBadge
import com.personal.driverguard.views.components.SectionHeader
import com.personal.driverguard.views.components.SkyScaffold
import com.personal.driverguard.views.components.plus
import com.personal.driverguard.views.components.alertLevelLabel
import com.personal.driverguard.views.components.endReasonLabel
import com.personal.driverguard.views.components.eventTypeLabel
import com.personal.driverguard.views.components.interruptionLabel
import com.personal.driverguard.views.components.ruleLabel
import com.personal.driverguard.views.components.formatDateTime
import com.personal.driverguard.views.components.formatDistance
import com.personal.driverguard.views.components.formatDuration
import com.personal.driverguard.views.components.formatTime
import com.personal.driverguard.views.components.groupedRow
import com.personal.driverguard.views.components.modeLabel
import com.personal.driverguard.views.container.TripDetailRoute
import com.personal.driverguard.views.theme.TabularNumbers
import com.personal.driverguard.views.theme.spacing
import com.personal.driverguard.views.theme.status
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.flow.stateIn
import java.util.Locale

sealed interface TripDetailState {
    data object Loading : TripDetailState
    data object NotFound : TripDetailState
    data class Loaded(val detail: TripDetailVo) : TripDetailState
    data class Failed(val message: String?) : TripDetailState
}

@OptIn(ExperimentalCoroutinesApi::class)
class TripDetailViewModel(savedStateHandle: SavedStateHandle, repository: ITripRepository) : ViewModel() {
    private val tripId = savedStateHandle.toRoute<TripDetailRoute>().tripId
    private val reload = MutableStateFlow(0)

    val state: StateFlow<TripDetailState> = reload
        .flatMapLatest {
            repository.observeTripDetail(tripId)
                .map { it?.let(TripDetailState::Loaded) ?: TripDetailState.NotFound }
                .onStart { emit(TripDetailState.Loading) }
                .catch { emit(TripDetailState.Failed(it.message)) }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), TripDetailState.Loading)

    fun retry() {
        reload.value++
    }
}

/** 여정 리포트(정차 상태에서 보는 화면). 가로: 왼쪽 요약, 오른쪽 이벤트 타임라인. */
@Composable
fun TripDetailScreen(onBack: () -> Unit, viewModel: TripDetailViewModel = appViewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    SkyScaffold(topBar = { GlassTopBar(stringResource(R.string.title_trip_detail), onBack) }) { padding ->
        val modifier = Modifier.fillMaxSize().padding(padding)
        when (val s = state) {
            TripDetailState.Loading -> LoadingState(stringResource(R.string.loading), modifier)
            TripDetailState.NotFound -> MessageState(
                icon = Icons.Rounded.Search,
                title = stringResource(R.string.trip_not_found),
                body = null,
                actionLabel = stringResource(R.string.action_back),
                onAction = onBack,
                modifier = modifier,
            )
            is TripDetailState.Failed -> MessageState(
                icon = Icons.Rounded.Warning,
                iconTint = MaterialTheme.colorScheme.error,
                title = stringResource(R.string.trip_error_title),
                body = stringResource(R.string.error_storage_body),
                actionLabel = stringResource(R.string.action_retry),
                onAction = viewModel::retry,
                modifier = modifier,
            )
            is TripDetailState.Loaded -> {
                val summary = Blocks().apply { summary(s.detail) }.list
                val timeline = Blocks().apply { timeline(s.detail) }.list
                // 바 여백은 스크롤 안쪽에 둬서 내용이 떠 있는 상단 바 아래로 흘러 들어간다
                AdaptiveLayout(
                    modifier = Modifier.fillMaxSize(),
                    // 가로: 요약과 타임라인 두 열이 스크롤 하나로 함께 움직인다(열마다 따로 스크롤하지 않는다)
                    wide = {
                        Row(
                            Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(padding.plus(MaterialTheme.spacing.m)),
                            horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m),
                        ) {
                            Column(Modifier.weight(1f)) { summary.forEach { (_, block) -> block() } }
                            Column(Modifier.weight(1.15f)) { timeline.forEach { (_, block) -> block() } }
                        }
                    },
                    compact = {
                        LazyColumn(Modifier.fillMaxSize(), contentPadding = padding.plus(MaterialTheme.spacing.m)) {
                            (summary + timeline).forEach { (key, block) -> item(key = key) { block() } }
                        }
                    },
                )
            }
        }
    }
}

/** 리포트 구성 블록. 세로 화면은 LazyColumn 항목으로, 가로 화면은 두 열에 그대로 그린다. */
private class Blocks {
    val list = mutableListOf<Pair<String, @Composable () -> Unit>>()

    fun item(key: String, content: @Composable () -> Unit) {
        list += key to content
    }
}

private fun Blocks.summary(detail: TripDetailVo) {
    val trip = detail.trip
    item(key = "header") {
        Column(verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s)) {
            if (trip.mode == AnalysisMode.DEMO) DemoNotice(stringResource(R.string.demo_banner_report))
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.s)) {
                // 시연 기록은 위의 띠가 이미 알리므로 배지는 실제 분석일 때만 붙인다
                if (trip.mode == AnalysisMode.REAL) ModeBadge(trip.mode)
                Text(formatDateTime(trip.startedAtMs), style = MaterialTheme.typography.titleMedium.merge(TabularNumbers), color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Column(Modifier.semantics(mergeDescendants = true) {}) {
                Text(stringResource(R.string.detail_duration), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(
                    trip.durationMs?.let(::formatDuration) ?: stringResource(R.string.trip_in_progress),
                    style = MaterialTheme.typography.displaySmall.merge(TabularNumbers),
                )
            }
        }
    }
    item(key = "facts") {
        InsetGroup(Modifier.padding(top = MaterialTheme.spacing.l)) {
            Column(Modifier.padding(MaterialTheme.spacing.m), verticalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.m)) {
                Row(horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.l)) {
                    KeyValue(stringResource(R.string.detail_started), formatTime(trip.startedAtMs), Modifier.weight(1f))
                    KeyValue(stringResource(R.string.detail_ended), trip.endedAtMs?.let(::formatTime) ?: stringResource(R.string.trip_in_progress), Modifier.weight(1f))
                    KeyValue(
                        stringResource(R.string.detail_distance),
                        trip.distanceMeters?.let(::formatDistance) ?: stringResource(R.string.detail_distance_none),
                        Modifier.weight(1f),
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.l)) {
                    KeyValue(stringResource(R.string.detail_valid_observation), trip.validObservationMs?.let(::formatDuration) ?: stringResource(R.string.unknown), Modifier.weight(1f))
                    KeyValue(stringResource(R.string.detail_unavailable), trip.unavailableMs?.let(::formatDuration) ?: stringResource(R.string.unknown), Modifier.weight(1f))
                    // 첫 줄(세 칸)과 칸을 맞춘다
                    Spacer(Modifier.weight(1f))
                }
                if (trip.endedAbnormally) {
                    Text(stringResource(R.string.trip_abnormal_detail), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                }
                Text(
                    stringResource(
                        R.string.detail_analyzer_value,
                        listOfNotNull(trip.modelInfo, stringResource(R.string.detail_config_version, trip.configVersion), trip.calibrationSummary).joinToString(" · "),
                    ),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
    for ((key, judgments) in listOf("judgments" to true, "indicators" to false)) item(key = key) {
        val summary = summarizeEvents(detail.events.filter { it.type.isJudgment == judgments })
        InsetGroup(Modifier.padding(top = MaterialTheme.spacing.l), header = stringResource(if (judgments) R.string.detail_judgments else R.string.detail_indicators)) {
            if (summary.isEmpty()) GroupRow(stringResource(R.string.detail_no_events))
            summary.forEachIndexed { index, it ->
                val unknown = if (it.unknownDurationCount > 0) stringResource(R.string.detail_unknown_duration, it.unknownDurationCount) else ""
                GroupRow(
                    title = stringResource(eventTypeLabel(it.type)),
                    subtitle = stringResource(R.string.detail_summary_count, it.count) + unknown,
                    trailing = { Text(formatDuration(it.totalDurationMs), style = MaterialTheme.typography.titleMedium.merge(TabularNumbers)) },
                    showDivider = index > 0,
                )
            }
        }
    }
    item(key = "rests") {
        InsetGroup(Modifier.padding(top = MaterialTheme.spacing.l), header = stringResource(R.string.detail_rests)) {
            if (detail.rests.isEmpty()) GroupRow(stringResource(R.string.detail_no_rests))
            detail.rests.forEachIndexed { index, rest ->
                val end = rest.endedAtMs?.let(::formatTime) ?: "-"
                GroupRow(
                    title = "${formatTime(rest.startedAtMs)} ~ $end",
                    trailing = {
                        Text(rest.durationMs?.let(::formatDuration) ?: stringResource(R.string.unknown), style = MaterialTheme.typography.titleMedium.merge(TabularNumbers))
                    },
                    showDivider = index > 0,
                )
            }
        }
    }
    item(key = "interruptions") {
        if (detail.interruptions.isEmpty()) return@item
        InsetGroup(Modifier.padding(top = MaterialTheme.spacing.l), header = stringResource(R.string.detail_interruptions)) {
            detail.interruptions.forEachIndexed { index, it ->
                GroupRow(
                    title = stringResource(interruptionLabel(it.reason)),
                    subtitle = "${formatTime(it.startedAtMs)} ~ ${it.endedAtMs?.let(::formatTime) ?: "-"}",
                    trailing = { Text(it.durationMs?.let(::formatDuration) ?: stringResource(R.string.unknown), style = MaterialTheme.typography.titleMedium.merge(TabularNumbers)) },
                    showDivider = index > 0,
                )
            }
        }
    }
    item(key = "located") {
        val located = detail.events.filter { it.latitude != null && it.longitude != null }
        InsetGroup(Modifier.padding(top = MaterialTheme.spacing.l), header = stringResource(R.string.detail_located_events)) {
            if (located.isEmpty()) GroupRow(stringResource(R.string.detail_no_located_events))
            located.forEachIndexed { index, e ->
                GroupRow(
                    title = stringResource(eventTypeLabel(e.type)) + " · " + formatTime(e.confirmedAtMs),
                    subtitle = String.format(Locale.US, "%.5f, %.5f", e.latitude, e.longitude) + (e.locationAccuracyMeters?.let { " (±${it.toInt()}m)" } ?: ""),
                    showDivider = index > 0,
                )
            }
        }
    }
}

private fun Blocks.timeline(detail: TripDetailVo) {
    item(key = "eventsHeader") {
        SectionHeader(stringResource(R.string.detail_events), Modifier.padding(start = MaterialTheme.spacing.m, top = MaterialTheme.spacing.l, bottom = MaterialTheme.spacing.xs))
    }
    if (detail.events.isEmpty()) {
        item(key = "eventsEmpty") { Column(Modifier.groupedRow(0, 1)) { GroupRow(stringResource(R.string.detail_no_events)) } }
    }
    detail.events.forEachIndexed { index, event ->
        item(key = "event-${event.id}") {
            Column(Modifier.groupedRow(index, detail.events.size)) { EventRow(event, showDivider = index > 0) }
        }
    }
}

@Composable
private fun EventRow(event: DriverEventVo, showDivider: Boolean) {
    val duration = event.durationMs?.let(::formatDuration) ?: stringResource(R.string.unknown)
    val rules = event.rules.split(",").filter { it.isNotBlank() }.map { id -> ruleLabel(id)?.let { r -> stringResource(r) } ?: id }.joinToString(", ")
    val meta = listOfNotNull(
        listOfNotNull(
            stringResource(alertLevelLabel(event.level)),
            stringResource(modeLabel(event.source)),
            event.endReason?.let { stringResource(endReasonLabel(it)) } ?: stringResource(R.string.event_in_progress),
        ).joinToString(" · "),
        stringResource(R.string.event_candidate, formatTime(event.candidateStartedAtMs), formatTime(event.confirmedAtMs)),
        rules.takeIf { it.isNotEmpty() }?.let { stringResource(R.string.event_rules, it) },
        event.metrics.takeIf { it.isNotEmpty() }?.let { stringResource(R.string.event_metrics, it) },
    ).joinToString("\n")
    GroupRow(
        title = stringResource(eventTypeLabel(event.type)),
        subtitle = meta,
        modifier = Modifier.semantics(mergeDescendants = true) {},
        leading = {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(MaterialTheme.spacing.xs)) {
                // 이벤트 단계를 그때의 지평선 빛 색으로 기억한다(문구로도 함께 표시)
                Box(Modifier.size(MaterialTheme.spacing.xs).background(levelHue(event.level), CircleShape))
                Text(formatTime(event.confirmedAtMs), style = MaterialTheme.typography.bodyMedium.merge(TabularNumbers), color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        // 지속 시간은 오른쪽 정렬 고정폭 숫자로 줄을 맞춘다
        trailing = { Text(duration, style = MaterialTheme.typography.titleMedium.merge(TabularNumbers)) },
        showDivider = showDivider,
    )
}

@Composable
private fun levelHue(level: AlertLevel) = when (level) {
    AlertLevel.INFO -> MaterialTheme.status.slate
    AlertLevel.CAUTION -> MaterialTheme.status.dawn
    AlertLevel.REST -> MaterialTheme.status.sunset
}

