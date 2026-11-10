package com.personal.driverguard.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.personal.driverguard.data.remote.SessionEvent
import com.personal.driverguard.ui.design.DesignBoard
import com.personal.driverguard.ui.state.HawkeyeViewModel
import com.personal.driverguard.ui.screens.*

/** Every screen of the design, in the order of the review board. */
enum class Screen(val title: String) {
    Splash("스플래시"), Login("로그인"), LoginError("로그인 실패"), Signup("회원가입, 비밀번호 불일치"), SignupEmpty("회원가입, 필수 항목 누락"),
    Terms("약관 동의"), Home("홈"),
    Setup("보는 방향 설정, 설정 전"), SetupDone("보는 방향 설정, 설정 완료"), SetupNoFace("보는 방향 설정, 얼굴 확인 불가"),
    Observe("관찰 중"), ObserveLook("전방 주시 경고"), ObserveDrowsy("졸음 경고"),
    Suggest("경유지 추가 제안, 졸음 후"), SuggestRepeat("경유지 추가 제안, 부주의 3회"), SuggestExisting("기존 경유지가 있을 때"),
    SuggestNoDest("안내 중인 목적지 없음"), SuggestFull("경유지가 가득 참"), SuggestFail("내비 연동 실패"), SuggestAdded("경유지 추가 완료"),
    SuggestNoPlace("쉼터를 찾지 못함"),
    EndConfirm("관찰 종료 확인"), Report("이번 주행 리포트"), ReportFromHistory("주행 리포트, 기록에서 열기"),
    HistoryDay("일간 리포트"), HistoryCal("날짜 선택"), HistoryDayNone("일간 리포트, 주행 기록 없음"),
    HistoryWeek("주간 통계"), HistoryWeekEmpty("주간 통계, 졸음 0회"),
    Gallery("화면 목록"),
}

/** State that survives moving between screens (the brief: terms keep their checks after 전문 보기). */
class AppState(preview: Boolean) {
    /** 시안 예시 값 상태에서 쓰는 약관 선택(실제 약관은 HawkeyeViewModel.terms) */
    val terms = mutableStateListOf(false, true, true, true, false, false) // all, service, privacy, location, face, alerts(optional)

    /**
     * 시안 확인 모드. 화면 목록(스플래시 길게 누르기)이나 스크린샷 테스트로 연 화면은 시안의 예시 값을 그대로 보여 주고,
     * 앱을 정상 실행해 들어간 화면은 백엔드 데이터를 보여 준다.
     */
    var preview by mutableStateOf(preview)
}

/** 관찰 중에 보이는 화면(카메라 카드가 있는 화면) */
private val OBSERVING_SCREENS = setOf(
    Screen.Observe, Screen.ObserveLook, Screen.ObserveDrowsy, Screen.Suggest, Screen.SuggestRepeat, Screen.SuggestExisting,
    Screen.SuggestNoDest, Screen.SuggestFull, Screen.SuggestFail, Screen.SuggestAdded, Screen.SuggestNoPlace,
)

/** 화면에서 쓰는 앱 상태(백엔드·차량 내비 호출) */
val LocalHawkeye = staticCompositionLocalOf<HawkeyeViewModel> { error("HawkeyeViewModel 이 없습니다") }

class Navigator(start: Screen) {
    val stack = mutableStateListOf(start)
    val current get() = stack.last()
    fun go(s: Screen) { if (s == Screen.Home) { stack.clear() }; stack.add(s) }
    /** 스택을 비우고 [s] 하나만 남긴다(로그아웃·세션 만료) */
    fun reset(s: Screen) { stack.clear(); stack.add(s) }
    fun replace(s: Screen) { stack.removeAt(stack.lastIndex); stack.add(s) }
    fun back(): Boolean { if (stack.size > 1) { stack.removeAt(stack.lastIndex); return true }; return false }
}

@Composable
fun HawkeyeApp(start: Screen = Screen.Splash) {
    val nav = remember { Navigator(start) }
    val state = remember { AppState(preview = start != Screen.Splash || com.personal.driverguard.ui.state.DebugFlags.CLICK_THROUGH) }
    val vm: HawkeyeViewModel = viewModel()
    BackHandler(enabled = nav.stack.size > 1) { nav.back() }
    // 로그인 만료 → 로그인 화면, 약관 미동의 → 약관 동의 화면
    LaunchedEffect(vm) {
        vm.sessionEvents.collect { e ->
            if (state.preview) return@collect
            when (e) {
                SessionEvent.LOGGED_OUT -> nav.reset(Screen.Login)
                SessionEvent.TERMS_REQUIRED -> if (nav.current != Screen.Terms) nav.go(Screen.Terms)
            }
        }
    }
    // 테스트용 자동 경고: 관찰 화면에 있을 때만 해당 경고 화면으로 바꾼다(종료 확인 등 다른 화면이면 건너뜀)
    LaunchedEffect(vm) {
        vm.demoCues.collect { cue ->
            if (nav.current !in OBSERVING_SCREENS) return@collect
            nav.replace(if (cue == com.personal.driverguard.ui.state.DemoCue.DROWSY) Screen.ObserveDrowsy else Screen.Suggest)
        }
    }
    CompositionLocalProvider(LocalHawkeye provides vm) { DesignBoard {
        AnimatedContent(nav.current, transitionSpec = { fadeIn(tween(180)) togetherWith fadeOut(tween(120)) }, label = "screen") { s ->
            ScreenContent(s, nav, state)
        }
    } }
}

@Composable
fun ScreenContent(s: Screen, nav: Navigator, state: AppState) {
    when (s) {
        Screen.Splash -> SplashScreen(nav, state)
        Screen.Login -> LoginScreen(nav, state, error = false)
        Screen.LoginError -> LoginScreen(nav, state, error = true)
        Screen.Signup -> SignupScreen(nav, state, empty = false)
        Screen.SignupEmpty -> SignupScreen(nav, state, empty = true)
        Screen.Terms -> TermsScreen(nav, state)
        Screen.Home -> HomeScreen(nav, state)
        Screen.Setup -> SetupScreen(nav, state, SetupState.Before)
        Screen.SetupDone -> SetupScreen(nav, state, SetupState.Done)
        Screen.SetupNoFace -> SetupScreen(nav, state, SetupState.NoFace)
        Screen.Observe -> ObserveScreen(nav, state)
        Screen.ObserveLook -> LookScreen(nav, state)
        Screen.ObserveDrowsy -> DrowsyScreen(nav, state)
        Screen.Suggest, Screen.SuggestRepeat, Screen.SuggestExisting, Screen.SuggestNoDest,
        Screen.SuggestFull, Screen.SuggestFail, Screen.SuggestAdded, Screen.SuggestNoPlace -> SuggestScreen(nav, state, s)
        Screen.EndConfirm -> EndConfirmScreen(nav, state)
        Screen.Report -> ReportScreen(nav, state, fromHistory = false)
        Screen.ReportFromHistory -> ReportScreen(nav, state, fromHistory = true)
        Screen.HistoryDay -> HistoryDayScreen(nav, state, calendar = false)
        Screen.HistoryCal -> HistoryDayScreen(nav, state, calendar = true)
        Screen.HistoryDayNone -> HistoryDayNoneScreen(nav, state)
        Screen.HistoryWeek -> HistoryWeekScreen(nav, state, empty = false)
        Screen.HistoryWeekEmpty -> HistoryWeekScreen(nav, state, empty = true)
        Screen.Gallery -> GalleryScreen(nav, state)
    }
}
