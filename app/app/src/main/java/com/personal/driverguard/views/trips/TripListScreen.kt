package com.personal.driverguard.views.trips

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.automirrored.rounded.List
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import com.personal.driverguard.R
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.domain.repository.ITripRepository
import com.personal.driverguard.entity.TripListItemVo
import com.personal.driverguard.views.base.appViewModel
import com.personal.driverguard.views.components.GlassTopBar
import com.personal.driverguard.views.components.GroupRow
import com.personal.driverguard.views.components.groupedRow
import com.personal.driverguard.views.components.LoadingState
import com.personal.driverguard.views.components.MessageState
import com.personal.driverguard.views.components.ModeBadge
import com.personal.driverguard.views.components.SkyScaffold
import com.personal.driverguard.views.components.plus
import com.personal.driverguard.views.components.formatDateTime
import com.personal.driverguard.views.components.formatDuration
import com.personal.driverguard.views.theme.sizes
import com.personal.driverguard.views.theme.spacing
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.flow.stateIn

sealed interface TripListState {
    data object Loading : TripListState
    data object Empty : TripListState
    data class Loaded(val trips: List<TripListItemVo>) : TripListState
    data class Failed(val message: String?) : TripListState
}

@OptIn(ExperimentalCoroutinesApi::class)
class TripListViewModel(private val repository: ITripRepository) : ViewModel() {

    private val reload = MutableStateFlow(0)

    val state: StateFlow<TripListState> = reload
        .flatMapLatest {
            repository.observeTrips()
                .map<List<TripListItemVo>, TripListState> { if (it.isEmpty()) TripListState.Empty else TripListState.Loaded(it) }
                .onStart { emit(TripListState.Loading) }
                .catch { emit(TripListState.Failed(it.message)) }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), TripListState.Loading)

    fun retry() {
        reload.value++
    }
}

@Composable
fun TripListScreen(
    onBack: () -> Unit,
    onOpenTrip: (Long) -> Unit,
    viewModel: TripListViewModel = appViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    SkyScaffold(topBar = { GlassTopBar(stringResource(R.string.title_trips), onBack) }) { padding ->
        // 목록은 바 여백을 스크롤 안쪽에 둬서 떠 있는 상단 바 아래로 흘러 들어간다
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
            when (val s = state) {
                TripListState.Loading -> LoadingState(stringResource(R.string.loading), Modifier.padding(padding))
                TripListState.Empty -> MessageState(
                    icon = Icons.AutoMirrored.Rounded.List,
                    title = stringResource(R.string.trips_empty_title),
                    body = stringResource(R.string.trips_empty_body),
                    actionLabel = stringResource(R.string.action_back_home),
                    onAction = onBack,
                    modifier = Modifier.padding(padding),
                )
                is TripListState.Failed -> MessageState(
                    icon = Icons.Rounded.Warning,
                    iconTint = MaterialTheme.colorScheme.error,
                    title = stringResource(R.string.trips_error_title),
                    body = stringResource(R.string.error_storage_body),
                    actionLabel = stringResource(R.string.action_retry),
                    onAction = viewModel::retry,
                    modifier = Modifier.padding(padding),
                )
                is TripListState.Loaded -> LazyColumn(
                    Modifier.widthIn(max = MaterialTheme.sizes.contentMaxWidth).fillMaxSize(),
                    contentPadding = padding.plus(MaterialTheme.spacing.m),
                ) {
                    itemsIndexed(s.trips, key = { _, item -> item.trip.id }) { index, item ->
                        TripRow(item, index, s.trips.size) { onOpenTrip(item.trip.id) }
                    }
                }
            }
        }
    }
}

/** 둥근 묶음 목록의 한 줄. 첫 줄·마지막 줄만 모서리를 둥글게 해서 한 덩어리로 보이게 한다. */
@Composable
private fun TripRow(item: TripListItemVo, index: Int, count: Int, onClick: () -> Unit) {
    val trip = item.trip
    val duration = trip.durationMs?.let(::formatDuration) ?: stringResource(R.string.trip_in_progress)
    Column(Modifier.groupedRow(index, count)) {
        GroupRow(
            title = formatDateTime(trip.startedAtMs),
            subtitle = stringResource(R.string.trip_row_summary, duration, item.eventCount) +
                if (trip.endedAbnormally) " · " + stringResource(R.string.trip_abnormal) else "",
            // 실제 분석이 기본이므로 시연 기록에만 표시를 붙인다
            leading = if (trip.mode == AnalysisMode.DEMO) ({ ModeBadge(trip.mode) }) else null,
            trailing = { Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant) },
            onClick = onClick,
            showDivider = index > 0,
        )
    }
}

