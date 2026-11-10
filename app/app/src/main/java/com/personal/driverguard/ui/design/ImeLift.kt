package com.personal.driverguard.ui.design

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.ime
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp

/**
 * 화면은 adjustNothing 이라 키보드가 올라와도 줄어들지 않는다. 대신 포커스된 입력란이 키보드에 가리면
 * 그만큼만 화면 전체를 위로 올린다([Field] 가 포커스를 받으면 자기 위치를 알린다).
 */
class ImeFocus internal constructor() {
    internal var field by mutableStateOf<LayoutCoordinates?>(null)
    internal var owner by mutableStateOf<Any?>(null)

    fun focus(key: Any, coords: LayoutCoordinates) { owner = key; field = coords }
    fun move(key: Any, coords: LayoutCoordinates) { if (owner === key) field = coords }
    fun blur(key: Any) { if (owner === key) { owner = null; field = null } }
}

val LocalImeFocus = staticCompositionLocalOf<ImeFocus?> { null }

/** 입력란 아래 여유(키보드와 입력란·안내 문구 사이) */
private val IME_MARGIN = 24.dp

@Composable
fun ImeLift(modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) {
    val focus = remember { ImeFocus() }
    val density = LocalDensity.current
    val imePx = WindowInsets.ime.getBottom(density)
    val viewH = LocalView.current.height
    var outer by remember { mutableStateOf<LayoutCoordinates?>(null) }
    var inner by remember { mutableStateOf<LayoutCoordinates?>(null) }

    val field = focus.field
    val o = outer; val i = inner
    val target = if (imePx > 0 && field != null && o != null && i != null && field.isAttached && o.isAttached && i.isAttached) {
        // 올리기 전 위치로 계산한다(inner 안의 좌표라 이미 올린 만큼은 상쇄된다)
        val bottomInInner = i.localPositionOf(field, Offset(0f, field.size.height.toFloat())).y
        val bottomInWindow = o.boundsInWindow().top + bottomInInner
        val visibleBottom = viewH - imePx - with(density) { IME_MARGIN.toPx() }
        (bottomInWindow - visibleBottom).coerceAtLeast(0f)
    } else 0f
    val shift by animateFloatAsState(target, tween(200), label = "imeLift")

    CompositionLocalProvider(LocalImeFocus provides focus) {
        Box(modifier.fillMaxSize().onGloballyPositioned { outer = it }) {
            Box(Modifier.fillMaxSize().graphicsLayer { translationY = -shift }.onGloballyPositioned { inner = it }, content = content)
        }
    }
}
