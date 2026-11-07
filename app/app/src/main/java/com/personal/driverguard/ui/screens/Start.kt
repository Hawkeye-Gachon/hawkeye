package com.personal.driverguard.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.R
import com.personal.driverguard.ui.AppState
import com.personal.driverguard.ui.LocalHawkeye
import com.personal.driverguard.ui.Navigator
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.ui.design.*
import com.personal.driverguard.ui.state.Next
import com.personal.driverguard.ui.theme.NColor
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState

/** [Next] → 화면 이동. 홈은 스택을 비운다. */
fun Navigator.goNext(next: Next) = when (next) {
    Next.HOME -> go(Screen.Home)
    Next.TERMS -> go(Screen.Terms)
    Next.LOGIN -> reset(Screen.Login)
}

// ------------------------------------------------------------------ splash
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun SplashScreen(nav: Navigator, state: AppState) {
    val vm = LocalHawkeye.current
    var progress by remember { mutableFloatStateOf(.38f) }
    LaunchedEffect(Unit) {
        val start = System.currentTimeMillis()
        // 진행 막대가 차는 동안 저장된 로그인을 확인한다
        val next = if (state.preview) null else async { vm.bootstrap() }
        while (progress < 1f) { delay(16); progress = (.38f + (System.currentTimeMillis() - start) / 2200f).coerceAtMost(1f) }
        delay(250)
        if (nav.current != Screen.Splash) return@LaunchedEffect
        if (next == null) nav.replace(Screen.Login) else nav.goNext(next.await())
    }
    Box(Modifier.fillMaxSize()) { FullCard(Modifier.glassBg().combinedClickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onLongClick = { state.preview = true; nav.go(Screen.Gallery) }, onClick = {})) {
        Image(painterResource(R.drawable.head_hero), null, Modifier.at(396.dp, 12.dp, 440.dp, 481.dp))
        Column(Modifier.at(0.dp, 512.dp, 1232.dp, 140.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Mark(size = 72.dp, textStyle = S.t(64, 72, 300, tracking = -.05), gap = 18.dp)
            T("졸음과 전방 주시 이탈을 살피고, 쉬어갈 곳을 안내합니다", S.body.copy(fontSize = androidx.compose.ui.unit.TextUnit(21f, androidx.compose.ui.unit.TextUnitType.Sp)))
        }
        Box(Modifier.at(496.dp, 693.dp, 240.dp, 3.dp).clip(RoundedCornerShape(2.dp)).background(NColor.fill2)) {
            Box(Modifier.fillMaxHeight().fillMaxWidth(progress).background(NColor.indigo))
        }
    }
}
}

// ------------------------------------------------------------------ login / signup: one centred card
@Composable
private fun BoxScope.AuthCard(width: Int, picHeight: Int, markWidth: Int, markTop: Int, overlay: @Composable BoxScope.() -> Unit = {}, content: @Composable ColumnScope.() -> Unit) {
    FullCard {
        Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
            Box(Modifier.width(width.dp).height(picHeight.dp).clipToBounds()) {
                Image(painterResource(R.drawable.obj_mark), null, Modifier.at(((width - markWidth) / 2).dp, markTop.dp, markWidth.dp, (markWidth * 1640 / 1500).dp))
            }
            Column(Modifier.width(width.dp), verticalArrangement = Arrangement.spacedBy(18.dp), content = content)
        }
        overlay()
    }
}

@Composable
private fun ColumnScope.AuthActions(content: @Composable ColumnScope.() -> Unit) =
    Column(Modifier.fillMaxWidth().padding(top = 14.dp), verticalArrangement = Arrangement.spacedBy(12.dp), content = content)

/** 입력란 오류 표시: 오류가 있으면 빨간 테두리와 사유 */
private fun fieldState(error: String?) = if (error != null) FieldState.Error else FieldState.Normal

@Composable
fun LoginScreen(nav: Navigator, state: AppState, error: Boolean) {
    val vm = LocalHawkeye.current
    val scope = rememberCoroutineScope()
    val form by vm.login.collectAsStateWithLifecycle()
    // 시안 확인 모드는 시안의 예시 값을, 실제 앱은 빈 입력란으로 시작한다
    var id by remember { mutableStateOf(if (state.preview) "haneul.kim@mail.com" else "") }
    var pw by remember { mutableStateOf(if (!state.preview) "" else if (error) "123456" else "drive1234") }
    val pwError = if (state.preview) "비밀번호가 맞지 않습니다. 대소문자를 확인하고 다시 입력해 주세요.".takeIf { error } else form.fieldErrors["password"] ?: form.message
    Box(Modifier.fillMaxSize()) {
        AuthCard(520, 230, 290, -36) {
            Column(Modifier.fillMaxWidth().padding(bottom = 6.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                T("앱이름", S.t(30, 36, 600, tracking = -.03))
                T("계정으로 로그인하면 지난 주행 기록이 이어집니다", S.small)
            }
            val idError = if (state.preview) null else form.fieldErrors["email"]
            Field("계정", id, { id = it }, placeholder = "name@mail.com", state = fieldState(idError), message = idError)
            Field("비밀번호", pw, { pw = it }, password = true, forceFocus = state.preview && !error,
                state = fieldState(pwError), message = pwError)
            AuthActions {
                Btn(if (form.busy) "로그인 중" else "로그인", Modifier.fillMaxWidth(), enabled = !form.busy) {
                    if (state.preview) nav.go(Screen.Home)
                    else scope.launch { vm.login(id, pw)?.let { nav.goNext(it) } }
                }
                Btn("회원가입", Modifier.fillMaxWidth(), Kind.Ghost) { vm.clearForms(); nav.go(Screen.Signup) }
            }
        }
    }
}

@Composable
fun SignupScreen(nav: Navigator, state: AppState, empty: Boolean) {
    val vm = LocalHawkeye.current
    val scope = rememberCoroutineScope()
    val form by vm.signup.collectAsStateWithLifecycle()
    val p = state.preview
    var name by remember { mutableStateOf(if (!p || empty) "" else "김하늘") }
    var mail by remember { mutableStateOf(if (!p) "" else if (empty) "haneul.kim@mail" else "haneul.kim@mail.com") }
    var pw by remember { mutableStateOf(if (p) "drive12345" else "") }
    var pw2 by remember { mutableStateOf(if (!p) "" else if (empty) "drive12345" else "drive1234") }
    var age by remember { mutableStateOf(p) }
    // 시안 확인 모드의 오류 문구는 시안 그대로, 실제 앱은 앱 검사와 서버 fieldErrors
    val errors: Map<String, String> = if (p) {
        if (empty) mapOf("name" to "이름을 입력해 주세요.", "email" to "이메일 형식이 아닙니다. 예: name@mail.com")
        else mapOf("passwordConfirm" to "비밀번호가 일치하지 않습니다.")
    } else form.fieldErrors
    Box(Modifier.fillMaxSize()) {
        AuthCard(720, 170, 210, -30, overlay = { BackBtn("로그인으로", Modifier.at(32.dp, 32.dp, 154.dp, 52.dp)) { nav.back() } }) {
            Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                T("회원가입", S.h1.copy(fontSize = 36.sp, lineHeight = 46.sp))
                T("가입 후 약관에 동의하면 관찰을 시작할 수 있습니다", S.small)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Box(Modifier.weight(1f)) { Field("이름", name, { name = it }, placeholder = "이름", state = fieldState(errors["name"]), message = errors["name"]) }
                Box(Modifier.weight(1f)) { Field("이메일", mail, { mail = it }, placeholder = "name@mail.com", state = fieldState(errors["email"]), message = errors["email"]) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Box(Modifier.weight(1f)) { Field("비밀번호", pw, { pw = it }, password = true, state = fieldState(errors["password"]), message = errors["password"] ?: "영문과 숫자를 섞어 8자 이상") }
                Box(Modifier.weight(1f)) { Field("비밀번호 확인", pw2, { pw2 = it }, password = true, state = fieldState(errors["passwordConfirm"]), message = errors["passwordConfirm"]) }
            }
            // 백엔드 가입 조건(만 14세 이상 확인). 시안에는 없는 항목이라 입력란 아래에 한 줄로 둔다.
            Row(
                Modifier.clip(RoundedCornerShape(12.dp)).clickable { age = !age }.padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                CheckBox(age)
                T("만 14세 이상입니다", S.t(17, 22, 500, NColor.ink2))
                T("필수", S.t(14, 20, 600, NColor.indigo))
                errors["ageConfirmed"]?.let { T(it, S.t(15, 20, color = NColor.dangerInk)) }
            }
            form.message?.takeIf { !p }?.let { T(it, S.t(15, 20, color = NColor.dangerInk)) }
            AuthActions {
                Btn(if (form.busy) "가입 중" else "회원가입", Modifier.fillMaxWidth(), enabled = !form.busy) {
                    if (p) nav.go(Screen.Terms)
                    else scope.launch { vm.signup(name, mail, pw, pw2, age)?.let { nav.goNext(it) } }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ terms: text on the left, consent on the right
private val termNames = listOf("서비스 이용약관" to true, "개인정보 수집·이용" to true, "위치정보 이용" to true, "얼굴 영상 분석" to true, "주행 리포트 알림" to false)
private val termTexts: List<List<Pair<String?, String>>> = listOf(
    listOf("제1조 목적" to "이 약관은 앱이름이 제공하는 운전자 상태 관찰, 경고, 쉼터 안내, 주행 리포트 서비스의 이용 조건을 정합니다.",
        "제2조 서비스의 한계" to "경고와 안내는 운전을 돕는 보조 수단입니다. 안전 운전의 책임은 운전자에게 있으며, 운전 중에는 화면 조작을 최소화해야 합니다."),
    listOf("수집 항목" to "이름, 이메일, 주행 기록(관찰 시작·종료 시각, 이동 거리, 감지된 이벤트의 시각·유형·지속 시간)",
        "이용 목적과 보관" to "주행 리포트와 주간 통계를 만드는 데 씁니다. 탈퇴하면 바로 지웁니다."),
    listOf(null to "쉼터를 찾고 이동 경로와 이벤트 위치를 기록하는 데 씁니다. 관찰 중에만 수집하며, 리포트의 핀은 도로의 위험도가 아니라 운전자 상태가 바뀐 위치를 뜻합니다."),
    listOf("분석 항목" to "눈 감김 비율, 눈을 감고 있던 시간, 하품, 머리 방향, 시선이 정면을 벗어난 시간. 결과는 추정이며 의학적 진단이 아닙니다.",
        "보관" to "영상과 사진은 저장하지 않고 태블릿 안에서만 분석합니다. 감지된 시각, 유형, 지속 시간, 위치만 주행 기록으로 남습니다."),
    listOf(null to "주간 통계와 반복되는 감지 양상을 알림으로 보내 드립니다. 동의하지 않아도 모든 기능을 쓸 수 있습니다."),
)

/** 약관 전문(마크다운)을 제목·문단으로 나눈다. 제목은 #, 강조 ** 는 지운다. */
private fun markdownBlocks(md: String): List<Pair<String?, String>> {
    val out = mutableListOf<Pair<String?, String>>()
    var heading: String? = null
    val para = StringBuilder()
    fun flush() {
        if (heading != null || para.isNotBlank()) out += heading to para.toString().trim()
        heading = null; para.clear()
    }
    md.lines().map { it.replace("**", "").trimEnd() }.forEach { line ->
        when {
            line.trimStart().startsWith("#") -> { flush(); heading = line.trimStart().trimStart('#').trim() }
            line.isBlank() -> if (para.isNotBlank()) { flush() }
            else -> { if (para.isNotEmpty()) para.append('\n'); para.append(line.trim().removePrefix("- ").let { if (line.trim().startsWith("- ")) "· $it" else it }) }
        }
    }
    flush()
    return out
}

@Composable
fun TermsScreen(nav: Navigator, state: AppState) {
    if (state.preview) PreviewTermsScreen(nav, state) else LiveTermsScreen(nav)
}

/** 실제 약관: 목록·전문은 백엔드, 동의는 사용자 계정에 저장 */
@Composable
private fun LiveTermsScreen(nav: Navigator) {
    val vm = LocalHawkeye.current
    val ui by vm.terms.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val list = rememberLazyListState()
    var hit by remember { mutableIntStateOf(-1) }
    LaunchedEffect(Unit) { vm.loadTerms() }
    LaunchedEffect(hit) { if (hit >= 0) { delay(1800); hit = -1 } }
    Box(Modifier.fillMaxSize()) {
        SceneCard(Modifier.background(NColor.frost)) {
            Column(Modifier.padding(start = 40.dp, end = 40.dp, top = 36.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                T("약관 동의", S.h1); T("아래 약관을 읽고 오른쪽에서 동의할 항목을 고르세요.", S.body)
            }
            LazyColumn(Modifier.at(0.dp, 150.dp, 688.dp, 602.dp).drawWithFade(), state = list, contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 48.dp)) {
                itemsIndexed(ui.items) { i, item ->
                    Column(
                        Modifier.fillMaxWidth()
                            .drawBehind { if (i > 0 && hit != i) drawRect(NColor.line, Offset(20.dp.toPx(), 0f), Size(size.width - 40.dp.toPx(), 1.dp.toPx())) }
                            .clip(RoundedCornerShape(20.dp)).background(if (hit == i) NColor.indigo.copy(alpha = .08f) else Color.Transparent)
                            .padding(start = 20.dp, end = 20.dp, top = 22.dp, bottom = 20.dp),
                    ) {
                        Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(bottom = 8.dp)) {
                            T(item.title, S.t(20, 28, 600)); T(if (item.required) "필수" else "선택", S.t(14, 20, 600, if (item.required) NColor.indigo else NColor.ink3))
                        }
                        val blocks = item.content?.let(::markdownBlocks)
                        if (blocks == null) T("전문을 불러오지 못했습니다.", S.t(16, 26, color = NColor.ink3))
                        else blocks.forEach { (h, p) ->
                            if (h != null && h != item.title) T(h, S.t(16, 24, 600), Modifier.padding(top = 12.dp, bottom = 2.dp))
                            if (p.isNotEmpty()) T(p, S.t(16, 26, color = NColor.ink2))
                        }
                    }
                }
            }
            if (ui.loading) T("약관을 불러오는 중입니다", S.body, Modifier.at(40.dp, 170.dp, 600.dp, 40.dp))
            ui.error?.takeIf { ui.items.isEmpty() }?.let { T(it, S.t(17, 24, color = NColor.dangerInk), Modifier.at(40.dp, 170.dp, 600.dp, 80.dp)) }
        }
        Sheet {
            SheetBody(top = 36.dp, gap = 0.dp, content = {
                T("동의할 항목", S.h2, Modifier.padding(bottom = 6.dp))
                val requiredCount = ui.items.count { it.required }
                T("필수 ${requiredCount}개에 모두 동의하면 시작할 수 있습니다", S.small, Modifier.padding(bottom = 14.dp))
                TermRow("전체 동의", null, ui.allOn, bold = true, onToggle = { vm.toggleAllTerms() })
                ui.items.forEachIndexed { i, item ->
                    TermRow(item.title, item.required, ui.checks[item.termsId] == true, last = i == ui.items.lastIndex, onToggle = { vm.toggleTerm(item.termsId) },
                        onMore = { hit = i; scope.launch { list.animateScrollToItem(i) } })
                }
                ui.error?.takeIf { ui.items.isNotEmpty() }?.let { T(it, S.t(15, 20, color = NColor.dangerInk), Modifier.padding(top = 12.dp)) }
            }, bottom = {
                Btn(if (ui.busy) "저장 중" else "동의하고 시작하기", Modifier.fillMaxWidth(), enabled = ui.requiredOk && !ui.busy) {
                    scope.launch { if (vm.agreeTerms()) nav.go(Screen.Home) }
                }
            })
        }
    }
}

/** 시안 확인 모드: 시안의 예시 약관 */
@Composable
private fun PreviewTermsScreen(nav: Navigator, state: AppState) {
    val checks = state.terms
    val allOn = (1..5).all { checks[it] }
    val reqOk = (1..4).all { checks[it] }
    val list = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var hit by remember { mutableIntStateOf(-1) }
    LaunchedEffect(hit) { if (hit >= 0) { delay(1800); hit = -1 } }
    Box(Modifier.fillMaxSize()) {
        SceneCard(Modifier.background(NColor.frost)) {
            Column(Modifier.padding(start = 40.dp, end = 40.dp, top = 36.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                T("약관 동의", S.h1); T("아래 약관을 읽고 오른쪽에서 동의할 항목을 고르세요.", S.body)
            }
            LazyColumn(Modifier.at(0.dp, 150.dp, 688.dp, 602.dp).drawWithFade(), state = list, contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 48.dp)) {
                itemsIndexed(termNames) { i, (name, req) ->
                    Column(
                        Modifier.fillMaxWidth()
                            .drawBehind { if (i > 0 && hit != i) drawRect(NColor.line, Offset(20.dp.toPx(), 0f), Size(size.width - 40.dp.toPx(), 1.dp.toPx())) }
                            .clip(RoundedCornerShape(20.dp)).background(if (hit == i) NColor.indigo.copy(alpha = .08f) else Color.Transparent)
                            .padding(start = 20.dp, end = 20.dp, top = 22.dp, bottom = 20.dp),
                    ) {
                        Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(bottom = 8.dp)) {
                            T(name, S.t(20, 28, 600)); T(if (req) "필수" else "선택", S.t(14, 20, 600, if (req) NColor.indigo else NColor.ink3))
                        }
                        termTexts[i].forEach { (h, p) ->
                            if (h != null) T(h, S.t(16, 24, 600), Modifier.padding(top = 12.dp, bottom = 2.dp))
                            T(p, S.t(16, 26, color = NColor.ink2))
                        }
                    }
                }
            }
        }
        Sheet {
            SheetBody(top = 36.dp, gap = 0.dp, content = {
                T("동의할 항목", S.h2, Modifier.padding(bottom = 6.dp))
                T("필수 4개에 모두 동의하면 시작할 수 있습니다", S.small, Modifier.padding(bottom = 14.dp))
                TermRow("전체 동의", null, allOn, bold = true, onToggle = { val v = !allOn; (1..5).forEach { checks[it] = v } })
                termNames.forEachIndexed { i, (n, req) ->
                    TermRow(n, req, checks[i + 1], last = i == termNames.lastIndex, onToggle = { checks[i + 1] = !checks[i + 1] },
                        onMore = { hit = i; scope.launch { list.animateScrollToItem(i) } })
                }
            }, bottom = { Btn("동의하고 시작하기", Modifier.fillMaxWidth(), enabled = reqOk) { nav.go(Screen.Home) } })
        }
    }
}

@Composable
private fun TermRow(name: String, required: Boolean?, checked: Boolean, bold: Boolean = false, last: Boolean = false, onToggle: () -> Unit, onMore: (() -> Unit)? = null) {
    Row(
        Modifier.fillMaxWidth().height(68.dp).drawBehind { if (!last) drawRect(NColor.line, Offset(0f, size.height - 1.dp.toPx()), Size(size.width, 1.dp.toPx())) },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(Modifier.weight(1f).fillMaxHeight().clickable(remember { MutableInteractionSource() }, null, onClick = onToggle), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            CheckBox(checked)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                T(name, if (bold) S.t(20, 26, 600) else S.t(18, 24))
                if (required != null) T(if (required) "필수" else "선택", S.t(14, 20, 600, if (required) NColor.indigo else NColor.ink3))
            }
        }
        if (onMore != null) Row(Modifier.clip(RoundedCornerShape(14.dp)).clickable(onClick = onMore).padding(start = 10.dp, end = 2.dp, top = 12.dp, bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            T("전문 보기", S.t(15, 20, color = NColor.ink3)); Ic(R.drawable.ic_right, NColor.ink3, 18.dp)
        }
    }
}

/** Fades the last 20% of a long document, like the design's mask. */
fun Modifier.drawWithFade(): Modifier = this
    .clipToBounds()
    .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
    .drawWithContent {
        drawContent()
        drawRect(Brush.verticalGradient(listOf(Color.Black, Color.Transparent), size.height * .8f, size.height), blendMode = BlendMode.DstIn)
    }
