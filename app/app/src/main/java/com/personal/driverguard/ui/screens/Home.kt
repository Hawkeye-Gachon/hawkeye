package com.personal.driverguard.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.data.remote.WeeklyReport
import com.personal.driverguard.ui.AppState
import com.personal.driverguard.ui.LocalHawkeye
import com.personal.driverguard.ui.state.Fmt
import com.personal.driverguard.ui.state.HomeUi
import java.time.LocalDate
import kotlinx.coroutines.launch
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.personal.driverguard.R
import com.personal.driverguard.ui.Navigator
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.ui.design.*
import com.personal.driverguard.ui.theme.NColor

/** Sample week used across home and history (Asia/Seoul, 11월 2일 ~ 8일). */
object Sample {
    val days = listOf("월", "화", "수", "목", "금", "토", "일")
    val weekIn = listOf<Int?>(3, 1, 2, 4, 4, null, null)
    val weekDr = listOf<Int?>(0, 0, 1, 0, 1, null, null)
    private val N: Int? = null
    val dayIn = listOf(N, N, N, N, N, N, N, N, 1, N, N, N, N, N, 3, 0, N, N, N, N, N, N, N, N)
    val dayDr = listOf(N, N, N, N, N, N, N, N, 0, N, N, N, N, N, 1, 0, N, N, N, N, N, N, N, N)
    val dayObs = mapOf(8 to 35, 14 to 29, 15 to 13)
    val hourIn = listOf(N, N, N, N, N, N, N, N, 4, 0, N, N, N, N, 3, 0, N, N, 5, 2, N, N, N, N)
    val hourDr = listOf(N, N, N, N, N, N, N, N, 0, 0, N, N, N, N, 1, 0, N, N, 0, 1, N, N, N, N)
    val weekObs = mapOf(8 to 175, 9 to 6, 14 to 29, 15 to 13, 18 to 130, 19 to 47)
    val hourLabel: (Int) -> String? = { i -> if (i % 3 == 0) "${i}시" else null }
}

/** 홈 화면에 보여 줄 값(시안 예시 또는 백엔드 데이터) */
private class HomeView(
    val userName: String?,
    /** null 이면 이전 기록 없음 */
    val lastLine: String?,
    val counts: List<Pair<String, String>>,
    val weekTitle: String,
    val weekLine: String,
    val weekValues: List<Int?>,
)

private val SampleHome = HomeView(
    userName = "김하늘",
    lastLine = "오늘 오후 2:30 출발, 관찰 42분 18초",
    counts = listOf("3회" to "부주의", "1회" to "졸음", "28.4km" to "이동"),
    weekTitle = "이번 주 부주의 14회",
    weekLine = "목요일과 금요일에 가장 많았습니다",
    weekValues = Sample.weekIn,
)

/** 요일별 부주의 값. 관찰하지 않은 날은 null(점선), 관찰했지만 0회면 0 */
fun weekInattention(w: WeeklyReport): List<Int?> = w.daily.map { if (it.observedSeconds > 0 || it.driveCount > 0) it.inattentionCount else null }

private fun liveHome(ui: HomeUi): HomeView {
    val last = ui.lastDrive
    val startedAt = Fmt.parse(last?.startedAt)
    val w = ui.weekly
    val values = w?.let(::weekInattention) ?: List(7) { null }
    val total = w?.summary?.inattentionCount ?: 0
    val max = values.maxOfOrNull { it ?: 0 } ?: 0
    val peakDays = values.indices.filter { values[it] == max && max > 0 }.map { Sample.days[it] + "요일" }
    return HomeView(
        userName = ui.me?.name,
        lastLine = last?.let { "${startedAt?.let { t -> Fmt.relativeDay(t.toLocalDate()) + " " + Fmt.time(t) } ?: ""} 출발, 관찰 ${Fmt.duration(it.observedSeconds)}" },
        counts = listOf("${last?.inattentionCount ?: 0}회" to "부주의", "${last?.drowsinessCount ?: 0}회" to "졸음", Fmt.distance(last?.distanceMeters) to "이동"),
        weekTitle = if (w == null) "이번 주 기록" else "${if (w.weekStart <= LocalDate.now().toString() && LocalDate.now().toString() <= w.weekEnd) "이번 주" else "최근 주"} 부주의 ${total}회",
        weekLine = when {
            ui.loading && w == null -> "불러오는 중입니다"
            w == null -> ui.error ?: "기록을 불러오지 못했습니다"
            peakDays.isEmpty() -> "이 기간에는 부주의가 감지되지 않았습니다"
            else -> peakDays.joinToString("과 ") + "에 가장 많았습니다"
        },
        weekValues = values,
    )
}

@Composable
fun HomeScreen(nav: Navigator, state: AppState) {
    val vm = LocalHawkeye.current
    val scope = rememberCoroutineScope()
    val ui by vm.home.collectAsStateWithLifecycle()
    LaunchedEffect(state.preview) { if (!state.preview) vm.loadHome() }
    val v = if (state.preview) SampleHome else liveHome(ui)
    Box(Modifier.fillMaxSize()) {
        SceneCard(Modifier.glassBg()) {
            Image(painterResource(R.drawable.head_hero), null, Modifier.at(160.dp, 52.dp, 370.dp, 405.dp))
            Row(Modifier.at(40.dp, 32.dp, 608.dp, 52.dp), verticalAlignment = Alignment.CenterVertically) {
                Mark(); Fill()
                Btn(v.userName?.let { "${it}님 로그아웃" } ?: "로그아웃", kind = Kind.Ghost, height = 52.dp, radius = 16.dp, style = S.t(18, 24, 600), horizontal = 20.dp) {
                    if (state.preview) nav.go(Screen.Login) else scope.launch { vm.logout(); nav.reset(Screen.Login) }
                }
            }
            Column(Modifier.align(Alignment.BottomStart).padding(44.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
                Column(Modifier.widthIn(max = 500.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    T("출발 전에 정면 방향을 설정해 주세요", S.h1)
                    T("평소 운전 자세로 앞유리 너머를 바라본 방향이 관찰 기준이 됩니다.", S.body)
                }
                Btn("관찰 시작", Modifier.fillMaxWidth(), height = 88.dp, radius = 28.dp, style = S.button.copy(fontSize = 26.sp, lineHeight = 32.sp)) { nav.go(Screen.Setup) }
            }
        }
        Sheet {
            SheetPic(R.drawable.scene_journey, height = 200.dp, biasY = .2f)
            SheetBody(gap = 12.dp, content = {
                T("이전 기록", S.h2)
                if (v.lastLine == null) {
                    T(if (ui.loading) "불러오는 중입니다" else "아직 주행 기록이 없습니다. 관찰을 시작하면 이곳에 쌓입니다.", S.body, Modifier.offset(y = (-6).dp))
                } else {
                    T(v.lastLine, S.body, Modifier.offset(y = (-6).dp))
                    Counts(v.counts)
                }
                Column(Modifier.padding(top = 10.dp)) {
                    HLine(); Gap(16.dp)
                    T(v.weekTitle, S.t(18, 26, 600))
                    T(v.weekLine, S.small)
                    Gap(6.dp)
                    BarChart(v.weekValues, { Sample.days[it] }, 440.dp, 128.dp, peak = peakOf(v.weekValues), padL = 0.dp, padT = 26.dp, padB = 30.dp, axis = false)
                }
            }, bottom = { Btn("이전 기록 보기", Modifier.fillMaxWidth(), Kind.Ghost) { nav.go(Screen.HistoryDay) } })
        }
    }
}
