package com.personal.driverguard.ui.state

/**
 * 쉼터로 가는 동안 졸지 않도록 음성으로 내는 간단한 운전 상식 O/X 퀴즈.
 * 문제와 답은 화면에 띄우지 않고 음성 안내(TTS)로 묻고 음성 인식(STT)으로 듣는다(운전 중 화면을 보지 않게).
 * Android 에 의존하지 않는다(단위 테스트로 검증).
 */
object DrowsyQuiz {
    /** [answer] 가 true 면 O, false 면 X. [fact] 는 틀렸거나 대답이 없을 때 알려 주는 한 문장. */
    data class Question(val text: String, val answer: Boolean, val fact: String)

    /** 운전·졸음운전 상식만 넣는다. 바로 답할 수 있게 한 문장으로 짧게 쓴다(생각하느라 운전에서 주의를 뺏지 않게). */
    val questions = listOf(
        Question("커피를 마시고 20분쯤 자면 잠이 더 잘 깬다.", true, "카페인은 20분쯤 뒤에 듣기 시작해서, 마시고 잠깐 자면 더 개운합니다."),
        Question("창문만 열면 졸음운전을 막을 수 있다.", false, "창문을 여는 효과는 잠깐뿐이라, 쉬어 가야 합니다."),
        Question("졸음쉼터는 졸릴 때 잠깐 쉬어 가는 곳이다.", true, "졸음쉼터는 졸릴 때 잠깐 쉬어 가는 곳입니다."),
        Question("고속도로에서 차가 고장 나면 차 안에서 기다린다.", false, "비상등을 켜고 가드레일 밖으로 피해야 합니다."),
        Question("두 시간 운전하면 15분쯤 쉬는 것이 좋다.", true, "두 시간마다 15분쯤 쉬는 것이 좋습니다."),
        Question("고속도로 1차로는 계속 달리는 차로다.", false, "1차로는 앞지르기할 때 쓰는 차로입니다."),
        Question("안전벨트는 뒷좌석에서도 매야 한다.", true, "안전벨트는 모든 좌석에서 매야 합니다."),
        Question("비 오는 날에는 제동 거리가 짧아진다.", false, "비가 오면 제동 거리가 길어집니다."),
        Question("감기약을 먹으면 졸릴 수 있다.", true, "일부 감기약과 알레르기약은 졸음을 부릅니다."),
        Question("운전 중에 휴대전화를 들고 통화해도 된다.", false, "운전 중에 휴대전화를 손에 들고 쓰면 안 됩니다."),
        Question("졸음운전은 음주운전만큼 위험하다.", true, "졸음운전은 반응이 늦어져 음주운전만큼 위험합니다."),
        Question("졸리면 빨리 달려서 일찍 도착하는 게 낫다.", false, "졸리면 속도를 내지 말고 가까운 쉼터에서 쉬어야 합니다."),
    )

    /** 처음 문제 앞에 붙이는 안내 */
    const val INTRO = "쉼터까지 졸지 않도록 간단한 운전 상식 퀴즈를 낼게요. 맞으면 오, 틀리면 엑스라고 답해 주세요."
    const val NEXT = "다음 문제입니다."

    private val X_WORDS = listOf("엑스", "엑쓰", "액스", "x", "틀", "아니", "거짓", "땡", "가위")
    private val O_WORDS = listOf("동그라미", "맞", "그래", "참", "정답")
    private val O_EXACT = setOf("오", "o", "5", "0", "ㅇ", "영", "네", "예", "응", "어", "오오", "오예")

    /**
     * 들은 말을 O(true)/X(false)로 본다. 판단할 수 없으면 null.
     * '아니오'처럼 X 를 뜻하는 말에 '오'가 들어 있을 수 있어 X 를 먼저 본다.
     */
    fun parse(heard: String?): Boolean? {
        val t = heard?.lowercase()?.filter { it.isLetterOrDigit() } ?: return null
        if (t.isEmpty()) return null
        if (X_WORDS.any { t.contains(it) }) return false
        if (t in O_EXACT || O_WORDS.any { t.contains(it) } || t.startsWith("오")) return true
        return null
    }

    /** 대답([said], 못 들었으면 null)에 대한 한마디 */
    fun feedback(q: Question, said: Boolean?): String {
        val right = if (q.answer) "오" else "엑스"
        return when (said) {
            null -> "대답을 듣지 못했어요. 정답은 ${right}입니다. ${q.fact}"
            q.answer -> "정답입니다!"
            else -> "아쉬워요. 정답은 ${right}입니다. ${q.fact}"
        }
    }
}
