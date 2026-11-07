package com.personal.driverguard.ui.screens

import androidx.annotation.DrawableRes
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.personal.driverguard.R
import com.personal.driverguard.domain.platform.NaviResult
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import android.os.SystemClock
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.TriggerType
import com.personal.driverguard.ui.AppState
import com.personal.driverguard.ui.LocalHawkeye
import com.personal.driverguard.ui.state.Fmt
import com.personal.driverguard.ui.state.HawkeyeViewModel
import com.personal.driverguard.ui.state.SuggestionUi
import com.personal.driverguard.ui.Navigator
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.ui.design.*
import com.personal.driverguard.ui.theme.NColor

// ------------------------------------------------------------------ camera scenes
private val CamPurple = Color(0xFF7B6BFF)
private val CamRed = Color(0xFFFF5A50)

@Composable
private fun CamLabel() = Row(Modifier.padding(start = 28.dp, top = 34.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
    Ic(R.drawable.ic_camera, NColor.onNight2, 20.dp); T("운전자 카메라, 적외선 30fps", S.t(16, 22, color = NColor.onNight2))
}

/** Full-face framing used while setting the forward direction. */
@Composable
private fun BoxScope.SetupCamera(state: SetupState) {
    SceneCard(Modifier.background(NColor.camBg)) {
        // 실제 전면 카메라. 권한이 없거나 카메라가 없으면 시안의 예시 사진을 보여 준다.
        DriverCamera(Modifier.fillMaxSize()) {
            val img = painterResource(R.drawable.ir_driver)
            if (state == SetupState.NoFace) {
                Image(img, null, Modifier.fillMaxSize().graphicsLayer { translationX = 290.dp.toPx(); translationY = 130.dp.toPx(); scaleX = 1.15f; scaleY = 1.15f; alpha = .55f }.blur(1.5.dp), contentScale = ContentScale.Crop, alignment = androidx.compose.ui.BiasAlignment(0f, -.4f))
            } else {
                Image(img, null, Modifier.fillMaxSize().graphicsLayer { alpha = .9f }, contentScale = ContentScale.Crop, alignment = androidx.compose.ui.BiasAlignment(0f, -.4f))
            }
        }
        Canvas(Modifier.fillMaxSize()) {
            val c = Offset(346.dp.toPx(), 270.dp.toPx()); val sz = Size(244.dp.toPx(), 312.dp.toPx())
            val tl = Offset(c.x - sz.width / 2, c.y - sz.height / 2)
            when (state) {
                SetupState.Before -> drawOval(Color.White.copy(alpha = .9f), tl, sz, style = Stroke(2.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(10.dp.toPx(), 10.dp.toPx()))))
                SetupState.NoFace -> drawOval(NColor.alert, tl, sz, style = Stroke(2.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(10.dp.toPx(), 10.dp.toPx()))))
                SetupState.Done -> {
                    drawOval(CamPurple, tl, sz, style = Stroke(4.dp.toPx()))
                    drawLine(CamPurple, c, Offset(c.x, 0f), 2.dp.toPx(), StrokeCap.Round, PathEffect.dashPathEffect(floatArrayOf(2.dp.toPx(), 8.dp.toPx())))
                }
            }
        }
        CamLabel()
        val (msg, col, w) = when (state) {
            SetupState.Before -> Triple("얼굴이 점선 안에 들어왔습니다", NColor.onNight2, 400)
            SetupState.Done -> Triple("기준 방향 등록됨, 좌우 2°, 상하 1° 이내", Color.White, 600)
            SetupState.NoFace -> Triple("얼굴이 화면 오른쪽 아래로 벗어났습니다", NColor.alert, 600)
        }
        T(msg, S.t(18, 24, w, col), Modifier.align(Alignment.BottomCenter).padding(bottom = 32.dp), align = TextAlign.Center)
    }
}

enum class CamMode { Ok, Look, Drowsy }

/** 관찰 경과 시간(초). 1초마다 다시 그린다. */
@Composable
private fun rememberElapsedSeconds(startedMonotonicMs: Long): Long {
    val elapsed by produceState(0L, startedMonotonicMs) {
        while (true) {
            value = ((SystemClock.elapsedRealtime() - startedMonotonicMs) / 1000).coerceAtLeast(0)
            delay(1_000)
        }
    }
    return elapsed
}

/** Close framing on the eyes; the lower half is the read-out area. 관찰 종료 sits top-right on every observing screen. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun BoxScope.ObserveCamera(nav: Navigator, state: AppState, mode: CamMode = CamMode.Ok) {
    var demo by remember { mutableStateOf(false) }
    SceneCard(Modifier.background(NColor.camBg).combinedClickable(remember { MutableInteractionSource() }, null, onLongClick = { demo = true }, onClick = {})) {
        // 실제 전면 카메라는 위쪽(아래 절반은 수치 영역이라 어둡게 덮인다)에만 그린다.
        // 권한이 없거나 카메라가 없으면 시안의 예시 사진을 보여 준다.
        DriverCamera(Modifier.at(0.dp, 0.dp, 688.dp, 395.dp)) {
            val img = painterResource(if (mode == CamMode.Drowsy) R.drawable.ir_driver_closed else R.drawable.ir_driver)
            Image(
                img, null, contentScale = ContentScale.FillBounds,
                colorFilter = if (mode == CamMode.Drowsy) ColorFilter.colorMatrix(ColorMatrix().apply { setToScale(.9f, .9f, .9f, 1f) }) else null,
                modifier = Modifier.at((-351).dp, (-40).dp, 1406.dp, 937.dp).then(
                    if (mode == CamMode.Look) Modifier.graphicsLayer { translationX = 150.dp.toPx(); rotationY = 28f; cameraDistance = 14f * density; transformOrigin = androidx.compose.ui.graphics.TransformOrigin(.6f, .3f); alpha = .9f } else Modifier
                ),
            )
        }
        Box(Modifier.fillMaxSize().background(Brush.verticalGradient(
            0f to Color(0x8C0A0B0D), .14f to Color(0x000A0B0D), .44f to Color(0x000A0B0D), .525f to Color(0xFB0A0B0D), 1f to NColor.camBg,
        )))
        Canvas(Modifier.fillMaxSize()) {
            fun px(v: Int) = v.dp.toPx()
            when (mode) {
                CamMode.Look -> {
                    val p = Path().apply { moveTo(px(380), px(160)); cubicTo(px(470), px(140), px(540), px(160), px(600), px(210)) }
                    drawPath(p, NColor.alert, style = Stroke(px(3), cap = StrokeCap.Round))
                    val h = Path().apply { moveTo(px(586), px(186)); lineTo(px(602), px(212)); lineTo(px(572), px(214)) }
                    drawPath(h, NColor.alert, style = Stroke(px(3), cap = StrokeCap.Round, join = StrokeJoin.Round))
                }
                else -> {
                    val c = if (mode == CamMode.Drowsy) CamRed else CamPurple
                    val ry = if (mode == CamMode.Drowsy) 14 else 18
                    val cy = if (mode == CamMode.Drowsy) 293 else 291
                    listOf(272, 419).forEach { cx -> drawOval(c, Offset(px(cx - 40), px(cy - ry)), Size(px(80), px(ry * 2)), style = Stroke(if (mode == CamMode.Drowsy) 2.5.dp.toPx() else px(2))) }
                    val br = Stroke(px(2), cap = StrokeCap.Round)
                    listOf(
                        listOf(226 to 228, 208 to 228, 208 to 246), listOf(466 to 228, 484 to 228, 484 to 246),
                        listOf(226 to 356, 208 to 356, 208 to 338), listOf(466 to 356, 484 to 356, 484 to 338),
                    ).forEach { pts -> drawPath(Path().apply { moveTo(px(pts[0].first), px(pts[0].second)); pts.drop(1).forEach { lineTo(px(it.first), px(it.second)) } }, c, style = br) }
                }
            }
        }
        CamLabel()
        Btn("관찰 종료", Modifier.align(Alignment.TopEnd).padding(top = 20.dp, end = 20.dp), Kind.OnDark, height = 52.dp, radius = 16.dp, style = S.t(18, 24, 600), horizontal = 22.dp) { nav.go(Screen.EndConfirm) }
        val (head, headColor, dot) = when (mode) {
            CamMode.Ok -> Triple("눈과 머리 방향을 추적하고 있습니다", NColor.onNight2, CamPurple)
            CamMode.Look -> Triple("머리가 기준 방향에서 벗어났습니다", NColor.alert, NColor.alert)
            CamMode.Drowsy -> Triple("눈을 감은 상태가 이어지고 있습니다", Color(0xFFFF7A72), CamRed)
        }
        Row(Modifier.align(Alignment.BottomStart).padding(start = 36.dp, bottom = 150.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Dot(dot, 8.dp); T(head, S.t(17, 24, color = headColor))
        }
        // 눈 뜬 정도·머리 방향 같은 수치는 AI 추론 서버가 붙으면 채운다. 실제 앱에서는 그 전까지 '–' 로 둔다.
        val tele = when (mode) {
            CamMode.Ok -> listOf(Triple("0.82", "눈 뜬 정도", Color.White), Triple("정면", "머리 방향, 기준에서 3° 이내", Color.White), Triple("4%", "최근 30초 눈 감김 비율", Color.White))
            CamMode.Look -> listOf(Triple("오른쪽 31°", "머리 방향", NColor.alert), Triple("2.4초", "이탈 지속 시간", NColor.alert), Triple("0.80", "눈 뜬 정도", Color.White))
            CamMode.Drowsy -> listOf(Triple("0.06", "눈 뜬 정도", CamRed), Triple("2.6초", "눈을 감고 있던 시간", CamRed), Triple("31%", "최근 30초 눈 감김 비율", CamRed))
        }.map { if (state.preview) it else Triple("–", it.second, it.third) }
        Row(Modifier.align(Alignment.BottomStart).padding(start = 36.dp, end = 36.dp, bottom = 36.dp).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
            tele.forEach { (v, l, c) ->
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    T(v, S.t(34, 40, 300, c, -.02).copy(fontFeatureSettings = "tnum")); T(l, S.t(15, 20, color = NColor.onNight3))
                }
            }
        }
        if (demo) DemoMenu(nav, state, mode) { demo = false }
    }
}

/**
 * 카메라 카드를 길게 누르면 여는 시연 신호. AI 추론 서버가 붙기 전까지 감지를 대신한다.
 * 실제 앱에서는 신호를 주행 기록에 남기고(HawkeyeViewModel.signal), 부주의 3회 누적 규칙도 그대로 적용한다.
 */
@Composable
private fun BoxScope.DemoMenu(nav: Navigator, state: AppState, mode: CamMode, close: () -> Unit) {
    val vm = LocalHawkeye.current
    val live = !state.preview
    fun inattention() { vm.signal(EventType.INATTENTION); nav.replace(Screen.ObserveLook) }
    fun drowsiness() { vm.signal(EventType.DROWSINESS); vm.openSuggestion(TriggerType.DROWSINESS); nav.replace(Screen.ObserveDrowsy) }
    /** 정면 주시 확인: 부주의가 3회 단위로 쌓였으면 쉼터 제안, 아니면 관찰로 돌아간다 */
    fun facingFront() {
        if (vm.inattentionDue) { vm.openSuggestion(TriggerType.INATTENTION_REPEATED); nav.replace(Screen.SuggestRepeat) } else nav.replace(Screen.Observe)
    }
    val items: List<Pair<String, () -> Unit>> = if (live) when (mode) {
        CamMode.Ok -> listOf("전방 주시 이탈" to ::inattention, "졸음 감지" to ::drowsiness)
        CamMode.Look -> listOf("정면 주시 확인" to ::facingFront, "졸음 동시 감지" to ::drowsiness)
        CamMode.Drowsy -> listOf("경고 후 휴식 제안" to { nav.replace(Screen.Suggest) })
    } else when (mode) {
        CamMode.Ok -> listOf("전방 주시 이탈" to { nav.replace(Screen.ObserveLook) }, "졸음 감지" to { nav.replace(Screen.ObserveDrowsy) }, "부주의 3회 누적" to { nav.replace(Screen.SuggestRepeat) })
        CamMode.Look -> listOf("정면 주시 확인" to { nav.replace(Screen.Observe) }, "세 번째 경고 뒤 정면 확인" to { nav.replace(Screen.SuggestRepeat) }, "졸음 동시 감지" to { nav.replace(Screen.ObserveDrowsy) })
        CamMode.Drowsy -> listOf("경고 후 휴식 제안" to { nav.replace(Screen.Suggest) })
    }
    Column(
        Modifier.align(Alignment.Center).width(360.dp).shadow(24.dp, RoundedCornerShape(24.dp)).clip(RoundedCornerShape(24.dp)).background(NColor.frost).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        T("시연 신호", S.t(15, 20, 600, NColor.ink3))
        items.forEach { (t, action) -> Btn(t, Modifier.fillMaxWidth(), Kind.Ghost, height = 56.dp, radius = 16.dp, style = S.t(18, 24, 600)) { close(); action() } }
        Btn("닫기", Modifier.fillMaxWidth(), Kind.Ghost, height = 48.dp, radius = 14.dp, style = S.t(16, 22, 600, NColor.ink3)) { close() }
    }
}

// ------------------------------------------------------------------ setup
enum class SetupState { Before, Done, NoFace }

@Composable
fun SetupScreen(nav: Navigator, state: AppState, setup: SetupState) {
    val vm = LocalHawkeye.current
    Box(Modifier.fillMaxSize()) {
        SetupCamera(setup)
        Sheet {
            SheetPic(R.drawable.head_setup, glass = true, headScale = true, biasX = -.52f) {
                BackBtn("홈", Modifier.padding(20.dp), Kind.OnImage) { nav.go(Screen.Home) }
            }
            SheetBody(content = {
                when (setup) {
                    SetupState.Before -> {
                        T("평소 운전 자세로 앉아 앞유리 너머 정면을 바라봐 주세요", S.h2)
                        T("화면을 보고 있으면 화면 쪽이 정면으로 등록됩니다. 차량 앞쪽을 바라본 채로 버튼을 눌러 주세요.", S.body)
                    }
                    SetupState.Done -> {
                        Ic(R.drawable.ic_check, NColor.indigo, 44.dp)
                        T("정면 방향을 설정했습니다", S.h2)
                        T("이 방향을 기준으로 시선과 얼굴 방향을 살펴봅니다. 자세를 크게 바꿨다면 다시 설정해 주세요.", S.body)
                    }
                    SetupState.NoFace -> {
                        Ic(R.drawable.ic_alert, NColor.alert, 44.dp)
                        T("얼굴이 보이도록 카메라 위치를 조정해 주세요", S.h2)
                        T("얼굴 전체가 점선 안에 들어오면 다시 설정할 수 있습니다. 모자나 마스크가 눈을 가리면 확인이 어렵습니다.", S.body)
                    }
                }
            }, bottom = {
                when (setup) {
                    SetupState.Before -> Btn("이 방향으로 설정", Modifier.fillMaxWidth()) { nav.replace(Screen.SetupDone) }
                    SetupState.Done -> {
                        Btn("관찰 시작", Modifier.fillMaxWidth()) {
                            // 주행을 등록하고 위치 기록을 시작한다(시안 확인 모드는 이동만)
                            if (!state.preview) vm.startDrive()
                            nav.go(Screen.Observe)
                        }
                        Btn("다시 설정", Modifier.fillMaxWidth(), Kind.Ghost) { nav.replace(Screen.Setup) }
                    }
                    SetupState.NoFace -> Btn("다시 설정", Modifier.fillMaxWidth()) { nav.replace(Screen.Setup) }
                }
            })
        }
    }
}

// ------------------------------------------------------------------ observing

/** 관찰 중 화면 값(시안 예시 또는 이번 주행) */
private class ObserveView(val clock: String, val inattention: Int, val drowsiness: Int)

@Composable
private fun rememberObserveView(state: AppState): ObserveView {
    if (state.preview) return ObserveView("00:18:42", 2, 0)
    val ui by LocalHawkeye.current.observe.collectAsStateWithLifecycle()
    val elapsed = if (ui.active) rememberElapsedSeconds(ui.startedMonotonicMs) else 0L
    return ObserveView(Fmt.clock(elapsed), ui.inattentionCount, ui.drowsinessCount)
}

@Composable
fun ObserveScreen(nav: Navigator, state: AppState) {
    val v = rememberObserveView(state)
    val vm = LocalHawkeye.current
    val ui by vm.observe.collectAsStateWithLifecycle()
    Box(Modifier.fillMaxSize()) {
        ObserveCamera(nav, state)
        Sheet {
            SheetPic(R.drawable.head_observe, glass = true, biasX = -.4f)
            SheetBody(gap = 10.dp, content = {
                StateLine("관찰 중")
                T(v.clock, S.clock)
                Box(Modifier.padding(top = 6.dp)) { Counts(listOf("${v.inattention}회" to "부주의", "${v.drowsiness}회" to "졸음")) }
            }, bottom = {
                InfoRows(
                    if (state.preview) listOf(
                        R.drawable.ic_route to boldParts("차량 내비가 " to false, "집" to true, "까지 안내 중입니다" to false),
                        R.drawable.ic_pin to boldParts("다음 쉼터는 " to false, "천안 졸음쉼터" to true, ", 전방 9.6km" to false),
                    ) else listOf(
                        R.drawable.ic_route to boldParts("졸음이 감지되면 " to false, "가까운 쉼터" to true, "를 차량 내비 경유지로 제안합니다" to false),
                        R.drawable.ic_pin to boldParts(if (ui.locationEnabled) "이동 경로와 감지 위치를 기록하고 있습니다" to false else "위치 권한이 없어 이동 경로와 쉼터 안내를 쓸 수 없습니다" to false),
                    ),
                )
            })
        }
    }
}

@Composable
fun LookScreen(nav: Navigator, state: AppState) {
    val v = rememberObserveView(state)
    // 시안 예시는 '3번째', 실제 앱은 이번 주행에서 몇 번째 이탈인지
    val nth = if (state.preview) 3 else v.inattention
    Box(Modifier.fillMaxSize()) {
        ObserveCamera(nav, state, CamMode.Look)
        Sheet {
            SheetPic(R.drawable.head_look, glass = true, biasX = -.4f)
            SheetBody(gap = 12.dp, content = {
                Ic(R.drawable.ic_alert, NColor.alert, 52.dp)
                T("전방을 봐 주세요", S.alarm)
                T("정면을 다시 바라보면 경고가 닫힙니다", S.body.copy(fontSize = 20.sp, lineHeight = 30.sp))
            }, bottom = {
                InfoRows(listOf(
                    R.drawable.ic_volume to boldParts("짧은 알림음과 함께 음성으로 안내합니다" to false),
                    R.drawable.ic_alert to boldParts("이번 주행 " to false, "${nth}번째" to true, " 전방 주시 이탈, 졸음 ${v.drowsiness}회" to false),
                ))
            })
        }
        EdgeGlow(Glow.Orange)
    }
}

@Composable
fun DrowsyScreen(nav: Navigator, state: AppState) {
    val vm = LocalHawkeye.current
    val sug by vm.suggestion.collectAsStateWithLifecycle()
    val placeRow = when {
        state.preview -> boldParts("가장 가까운 쉼터는 " to false, "천안 졸음쉼터" to true, ", 전방 4.2km" to false)
        sug.loading -> boldParts("가까운 쉼터를 찾고 있습니다" to false)
        sug.place != null -> sug.place!!.let { boldParts("가장 가까운 쉼터는 " to false, it.name to true, ", ${Fmt.ahead(it.roadDistanceMeters)}" to false) }
        else -> boldParts((sug.noPlaceReason ?: "안전한 곳에 정차해 주세요") to false)
    }
    Box(Modifier.fillMaxSize()) {
        ObserveCamera(nav, state, CamMode.Drowsy)
        Sheet {
            SheetPic(R.drawable.head_drowsy, glass = true, biasX = -.2f)
            Column(Modifier.weight(1f).fillMaxWidth().padding(start = 40.dp, end = 40.dp, top = 32.dp, bottom = 40.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Ic(R.drawable.ic_eye_closed, NColor.danger, 64.dp)
                Gap(12.dp)
                T("졸음이 감지됐습니다", S.alarm.copy(fontSize = 48.sp, lineHeight = 58.sp), align = TextAlign.Center)
                Gap(12.dp)
                T("가까운 안전한 곳에서 쉬어가세요", S.t(24, 34, color = NColor.ink2), align = TextAlign.Center)
                Spacer(Modifier.weight(1f))
                InfoRows(listOf(
                    R.drawable.ic_volume to boldParts("경고음과 음성 안내가 함께 나갑니다" to false),
                    R.drawable.ic_pin to placeRow,
                ))
            }
        }
        EdgeGlow(Glow.Red)
    }
}

// ------------------------------------------------------------------ waypoint suggestion and its outcomes

/** 제안 화면 문구에 쓰는 값(시안 예시 또는 백엔드가 찾은 쉼터·차량 내비 상태) */
private class SuggestView(
    val name: String,
    /** 졸음쉼터 / 휴게소 */
    val kind: String,
    /** 졸음쉼터, 전방 4.2km */
    val placeLine: String,
    val destination: String,
    val existing: List<String>,
)

private fun sampleSuggest(s: Screen) = if (s == Screen.SuggestRepeat) SuggestView("기흥휴게소", "휴게소", "휴게소, 전방 6.8km", "집", emptyList())
else SuggestView("천안 졸음쉼터", "졸음쉼터", "졸음쉼터, 전방 4.2km", "집", if (s == Screen.SuggestFull) listOf("오산 주유소", "천안아산역") else listOf("천안아산역"))

private fun liveSuggest(ui: SuggestionUi): SuggestView {
    val p = ui.place
    val kind = HawkeyeViewModel.placeTypeLabel(p?.type)
    return SuggestView(
        name = p?.name ?: "쉼터",
        kind = kind,
        placeLine = p?.let { "$kind, ${Fmt.ahead(it.roadDistanceMeters)}" } ?: "",
        destination = ui.navi?.destinationName ?: "목적지",
        existing = ui.navi?.waypointNames.orEmpty(),
    )
}

@Composable
fun SuggestScreen(nav: Navigator, state: AppState, s: Screen) {
    val vm = LocalHawkeye.current
    val ui by vm.suggestion.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val live = !state.preview
    val v = if (live) liveSuggest(ui) else sampleSuggest(s)

    // 쉼터 조회가 끝나면 결과와 차량 내비 상태에 맞는 화면으로 바꾼다.
    // Pleos 내비에 연결되지 않으면(일반 폰·에뮬레이터) 내비 상태 화면 전환은 건너뛴다.
    LaunchedEffect(s, ui.loading, live) {
        if (!live || ui.loading || (s != Screen.Suggest && s != Screen.SuggestRepeat)) return@LaunchedEffect
        val navi = ui.navi
        when {
            ui.place == null -> nav.replace(Screen.SuggestNoPlace)
            navi == null -> Unit
            !navi.routing -> nav.replace(Screen.SuggestNoDest)
            navi.waypointsFull -> nav.replace(Screen.SuggestFull)
            s == Screen.Suggest && navi.waypointNames.isNotEmpty() -> nav.replace(Screen.SuggestExisting)
        }
    }

    /** 차량 내비에 쉼터를 넣고 결과 화면으로 간다. 시안 확인 모드나 Pleos 내비가 없는 환경에서는 [demo] 화면으로 넘어간다. */
    fun request(destination: Boolean, demo: Screen, onSuccess: Screen) {
        if (!live) { nav.replace(demo); return }
        scope.launch {
            nav.replace(
                when (vm.addRestStop(destination)) {
                    null -> demo
                    NaviResult.Added -> onSuccess
                    NaviResult.NoDestination -> Screen.SuggestNoDest
                    NaviResult.Full -> Screen.SuggestFull
                    is NaviResult.Failed -> Screen.SuggestFail
                },
            )
        }
    }

    val ready = !live || (!ui.loading && !ui.busy && ui.place != null)
    val stops = v.existing.take(1).map { Stop.Was to it }
    val glow = when (s) { Screen.Suggest, Screen.SuggestExisting, Screen.SuggestNoDest, Screen.SuggestFull, Screen.SuggestFail, Screen.SuggestNoPlace -> Glow.RedFaint; else -> Glow.None }
    val grey = s == Screen.SuggestFull || s == Screen.SuggestFail || s == Screen.SuggestNoPlace
    Box(Modifier.fillMaxSize()) {
        ObserveCamera(nav, state)
        Sheet {
            SheetPic(R.drawable.scene_rest, biasX = .24f, biasY = .2f, tint = if (grey) Color(0x99808080) else null)
            SheetBody(content = {
                when (s) {
                    Screen.Suggest, Screen.SuggestRepeat -> if (live && ui.loading) {
                        T("가까운 쉼터를 찾고 있습니다", S.h2)
                        T("지금 차선에서 바로 들어갈 수 있는 휴게소·졸음쉼터를 확인하고 있습니다.", S.body)
                    } else {
                        T("${v.name}를 경유지로 추가할까요?", S.h2)
                        if (s == Screen.SuggestRepeat) T("이번 주행에서 전방 주시 이탈이 ${if (live) vm.observe.value.inattentionCount else 3}번 감지됐습니다.", S.body)
                        Place(v.placeLine)
                        Seq(listOf(Stop.Here to "현재 위치", Stop.Rest to v.name, Stop.Dest to v.destination), Modifier.padding(top = if (s == Screen.Suggest) 18.dp else 12.dp))
                    }
                    Screen.SuggestExisting -> {
                        T("${v.name}를 경유지로 추가할까요?", S.h2)
                        T("기존 경유지 ${v.existing.firstOrNull() ?: ""}보다 먼저 들르도록 넣습니다.", S.body); Place(v.placeLine)
                        Seq(listOf(Stop.Here to "현재 위치", Stop.Rest to v.name) + stops + listOf(Stop.Dest to v.destination), Modifier.padding(top = 12.dp))
                    }
                    Screen.SuggestNoDest -> {
                        T("${v.name}로 길 안내를 시작할까요?", S.h2); T("차량 내비에 안내 중인 목적지가 없어 쉼터를 목적지로 설정합니다.", S.body); Place(v.placeLine)
                        Seq(listOf(Stop.Here to "현재 위치", Stop.Rest to v.name), Modifier.padding(top = 12.dp))
                    }
                    Screen.SuggestFull -> {
                        Ic(R.drawable.ic_alert, NColor.alert, 40.dp)
                        T("현재 경로에 경유지를 더 추가할 수 없습니다", S.h2)
                        T("차량 내비에는 경유지를 2곳까지 넣을 수 있습니다. 정차한 뒤 내비에서 경유지를 정리하면 다시 추가할 수 있습니다.", S.body)
                        Seq(listOf(Stop.Here to "현재 위치") + v.existing.map { Stop.Was to it } + listOf(Stop.Dest to v.destination), Modifier.padding(top = 8.dp))
                    }
                    Screen.SuggestFail -> {
                        Ic(R.drawable.ic_alert, NColor.danger, 40.dp)
                        T("차량 내비에 경유지를 추가하지 못했습니다", S.h2)
                        T("내비가 응답하지 않았습니다. 관찰은 계속되고 있으니 잠시 후 다시 시도해 주세요.", S.body)
                        if (live) ui.failure?.let { T("사유: $it", S.small) }
                    }
                    Screen.SuggestNoPlace -> {
                        Ic(R.drawable.ic_alert, NColor.alert, 40.dp)
                        T("가까운 쉼터를 찾지 못했습니다", S.h2)
                        T(if (live) ui.noPlaceReason ?: "안전한 곳에 정차해 주세요." else "고속도로가 아닌 곳에서는 쉼터를 안내하지 않습니다. 안전한 곳에 정차해 주세요.", S.body)
                    }
                    else -> {
                        Ic(R.drawable.ic_check, NColor.indigo, 40.dp)
                        T("${v.name}를 경유지로 추가했습니다", S.h2)
                        T("지도와 길 안내는 차량 내비에서 이어집니다. 휴식 후에는 ${v.destination}까지 안내합니다.", S.body)
                        Seq(listOf(Stop.Here to "현재 위치", Stop.Rest to v.name, Stop.Dest to v.destination), Modifier.padding(top = 12.dp))
                    }
                }
            }, bottom = {
                when (s) {
                    Screen.Suggest, Screen.SuggestRepeat, Screen.SuggestExisting -> {
                        Btn(if (live && ui.busy) "추가하는 중" else "경유지 추가", Modifier.fillMaxWidth(), enabled = ready) {
                            request(destination = false, demo = Screen.SuggestAdded, onSuccess = Screen.SuggestAdded)
                        }
                        Btn("나중에", Modifier.fillMaxWidth(), Kind.Ghost) { nav.replace(Screen.Observe) }
                    }
                    Screen.SuggestNoDest -> {
                        Btn("쉼터로 길 안내", Modifier.fillMaxWidth(), enabled = ready) {
                            request(destination = true, demo = Screen.Observe, onSuccess = Screen.Observe)
                        }
                        Btn("나중에", Modifier.fillMaxWidth(), Kind.Ghost) { nav.replace(Screen.Observe) }
                    }
                    Screen.SuggestFull, Screen.SuggestNoPlace -> Btn("닫기", Modifier.fillMaxWidth(), Kind.Ghost) { nav.replace(Screen.Observe) }
                    Screen.SuggestFail -> {
                        Btn("다시 시도", Modifier.fillMaxWidth(), enabled = ready) {
                            request(destination = false, demo = Screen.SuggestAdded, onSuccess = Screen.SuggestAdded)
                        }
                        Btn("닫기", Modifier.fillMaxWidth(), Kind.Ghost) { nav.replace(Screen.Observe) }
                    }
                    else -> Btn("확인", Modifier.fillMaxWidth(), Kind.Ghost) { nav.replace(Screen.Observe) }
                }
            })
        }
        EdgeGlow(glow)
        if (s == Screen.Suggest && !live) SuggestVariants(nav)
    }
}

/** Small review switch under the sheet for the four navigation cases (design review only). */
@Composable
private fun BoxScope.SuggestVariants(nav: Navigator) {
    var open by remember { mutableStateOf(false) }
    Box(Modifier.at(736.dp, 24.dp, 520.dp, 60.dp).combinedClickableLong { open = true })
    if (open) Column(
        Modifier.align(Alignment.Center).width(380.dp).shadow(24.dp, RoundedCornerShape(24.dp)).clip(RoundedCornerShape(24.dp)).background(NColor.frost).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        T("차량 내비 상태", S.t(15, 20, 600, NColor.ink3))
        listOf("기존 경유지 있음" to Screen.SuggestExisting, "목적지 없음" to Screen.SuggestNoDest, "경유지 가득 참" to Screen.SuggestFull, "연동 실패" to Screen.SuggestFail)
            .forEach { (t, s) -> Btn(t, Modifier.fillMaxWidth(), Kind.Ghost, height = 56.dp, radius = 16.dp, style = S.t(18, 24, 600)) { open = false; nav.replace(s) } }
        Btn("닫기", Modifier.fillMaxWidth(), Kind.Ghost, height = 48.dp, radius = 14.dp, style = S.t(16, 22, 600, NColor.ink3)) { open = false }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun Modifier.combinedClickableLong(onLong: () -> Unit) = this.combinedClickable(remember { MutableInteractionSource() }, null, onLongClick = onLong, onClick = {})

@Composable
private fun Place(text: String) = Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    Ic(R.drawable.ic_pin, NColor.indigo); T(text, S.t(20, 28, color = NColor.ink2))
}

// ------------------------------------------------------------------ end confirm
@Composable
fun EndConfirmScreen(nav: Navigator, state: AppState) {
    val vm = LocalHawkeye.current
    val v = rememberObserveView(state)
    val end by vm.end.collectAsStateWithLifecycle()
    val obs by vm.observe.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val summary = if (state.preview) "18분 42초 동안 부주의 2회, 졸음 0회를 기록했습니다. 종료하면 이번 주행 리포트가 만들어집니다."
    else {
        val sec = if (obs.active) (SystemClock.elapsedRealtime() - obs.startedMonotonicMs) / 1000 else 0L
        "${Fmt.duration(sec.toInt())} 동안 부주의 ${v.inattention}회, 졸음 ${v.drowsiness}회를 기록했습니다. 종료하면 이번 주행 리포트가 만들어집니다."
    }
    Box(Modifier.fillMaxSize()) {
        ObserveCamera(nav, state)
        Sheet {
            SheetPic(R.drawable.head_observe, glass = true, biasX = -.4f)
            SheetBody(gap = 10.dp, content = { StateLine("관찰 중"); T(v.clock, S.clock) })
        }
        Scrim()
        Column(
            Modifier.at(340.dp, 150.dp, 600.dp, if (end.error != null) 450.dp else 390.dp).shadow(36.dp, CardShape).clip(CardShape).background(NColor.frost).padding(start = 48.dp, end = 48.dp, top = 44.dp, bottom = 48.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            if (end.error != null) {
                // 업로드 실패: 관찰은 이미 끝났고 기록은 기기에 남겨 두었다가 다음 실행 때 다시 올린다
                T("기록을 올리지 못했습니다", S.h2.copy(fontSize = 32.sp, lineHeight = 42.sp))
                T("${end.error} 이번 주행 기록은 기기에 저장했고, 앱을 다시 열면 자동으로 올립니다.", S.body.copy(fontSize = 19.sp, lineHeight = 29.sp))
                Gap(8.dp)
                Btn("홈으로", Modifier.fillMaxWidth()) { nav.go(Screen.Home) }
            } else {
                T("이번 주행의 관찰을 종료할까요?", S.h2.copy(fontSize = 32.sp, lineHeight = 42.sp))
                T(summary, S.body.copy(fontSize = 19.sp, lineHeight = 29.sp))
                Gap(8.dp)
                Btn(if (end.busy) "기록을 올리는 중" else "종료하고 리포트 보기", Modifier.fillMaxWidth(), enabled = !end.busy) {
                    if (state.preview) nav.go(Screen.Report)
                    else scope.launch { if (vm.endDrive()) nav.go(Screen.Report) }
                }
                Btn("계속 관찰하기", Modifier.fillMaxWidth(), Kind.Ghost, enabled = !end.busy) { nav.back() }
            }
        }
    }
}

@Suppress("unused") @DrawableRes private val keep = R.drawable.ic_mark
