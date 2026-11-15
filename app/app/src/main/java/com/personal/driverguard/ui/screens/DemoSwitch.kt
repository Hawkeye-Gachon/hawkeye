package com.personal.driverguard.ui.screens

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.ui.AppState
import com.personal.driverguard.ui.LocalHawkeye
import com.personal.driverguard.ui.Navigator
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.ui.design.S
import com.personal.driverguard.ui.design.T
import com.personal.driverguard.ui.theme.NColor
import kotlinx.coroutines.launch

/**
 * 시연 모드 켜기·끄기. 켜면 서버·차량 없이 모든 기능이 가짜로 동작한다(DemoMode).
 * 바꾸면 로그아웃하고 로그인 화면으로 간다(시연 계정과 실제 계정이 섞이지 않게).
 */
@Composable
fun DemoSwitch(nav: Navigator, state: AppState, modifier: Modifier = Modifier) {
    val vm = LocalHawkeye.current
    val on by vm.demoMode.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val knob by animateDpAsState(if (on) 22.dp else 2.dp, label = "knob")
    Row(
        modifier.clip(RoundedCornerShape(16.dp)).clickable {
            if (state.preview) return@clickable
            scope.launch {
                vm.setDemoMode(!on)
                nav.reset(Screen.Login)
            }
        }.padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        T("시연 모드", S.t(16, 22, 600, if (on) NColor.navy else NColor.ink3))
        Box(Modifier.width(44.dp).height(24.dp).clip(RoundedCornerShape(12.dp)).background(if (on) NColor.navy else NColor.line2)) {
            Box(Modifier.offset(x = knob, y = 2.dp).size(20.dp).clip(CircleShape).background(Color.White))
        }
    }
}
