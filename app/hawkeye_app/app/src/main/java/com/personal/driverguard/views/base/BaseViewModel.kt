package com.personal.driverguard.views.base

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.personal.driverguard.App
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow

/** 화면 이동 같은 일회성 이벤트를 [events] 로 흘려보낸다. 수신은 [EventEffect] 가 맡는다. */
abstract class BaseViewModel<E : Any> : ViewModel() {

    private val _events = Channel<E>(Channel.BUFFERED)
    val events: Flow<E> = _events.receiveAsFlow()

    protected fun sendEvent(event: E) {
        _events.trySend(event)
    }
}

/**
 * ViewModel 이벤트를 화면이 RESUMED 일 때만 처리한다.
 * 전환 애니메이션 중이거나 이미 떠나는 화면의 이벤트는 버려서, 연타로 화면이 두 번 열리는 일을 막는다.
 */
@Composable
fun <E> EventEffect(events: Flow<E>, onEvent: (E) -> Unit) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnEvent by rememberUpdatedState(onEvent)
    LaunchedEffect(events, lifecycleOwner) {
        events.collect { event ->
            if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) currentOnEvent(event)
        }
    }
}

/** [App.viewModelFactory] 로 ViewModel 을 만든다. 스코프는 현재 NavBackStackEntry. */
@Composable
inline fun <reified VM : ViewModel> appViewModel(): VM =
    viewModel(factory = (LocalContext.current.applicationContext as App).viewModelFactory)
