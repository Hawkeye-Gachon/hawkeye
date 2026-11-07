package com.personal.driverguard.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.personal.driverguard.ui.Navigator
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.ui.design.*
import com.personal.driverguard.ui.theme.NColor

/** Design review index: long-press the splash to open it. */
@Composable
fun GalleryScreen(nav: Navigator, state: com.personal.driverguard.ui.AppState) {
    // 화면 목록에서 여는 화면은 모두 시안의 예시 값으로 보여 준다
    androidx.compose.runtime.LaunchedEffect(Unit) { state.preview = true }
    Box(Modifier.fillMaxSize()) {
        FullCard {
            Column(Modifier.fillMaxSize().padding(start = 40.dp, end = 40.dp, top = 28.dp, bottom = 28.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    BackBtn("닫기") { nav.back() }; WGap(20.dp)
                    T("화면 목록", S.h1); Fill()
                    T("관찰 화면에서는 카메라 카드를 길게 누르면 시연 신호를 고를 수 있습니다", S.small)
                }
                LazyVerticalGrid(GridCells.Fixed(4), horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    itemsIndexed(Screen.entries.filter { it != Screen.Gallery }) { i, s ->
                        Row(
                            Modifier.height(64.dp).clip(RoundedCornerShape(16.dp)).background(NColor.fill).clickable { nav.go(s) }.padding(horizontal = 18.dp),
                            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            T("%02d".format(i + 1), S.t(15, 20, 600, NColor.ink3))
                            T(s.title, S.t(17, 22, 600), maxLines = 2)
                        }
                    }
                }
            }
        }
    }
}
