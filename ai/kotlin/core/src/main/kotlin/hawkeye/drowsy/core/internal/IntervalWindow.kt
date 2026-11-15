package hawkeye.drowsy.core.internal

// 원본: drowsy/rules.py:99-153 `IntervalWindow` (SPEC 12.1절). 계획 5.6절의 선언 그대로다.
// 시각과 길이는 모두 Long이다(R1). 두 목록은 kotlin.collections.ArrayDeque다(R13: removeFirst로 앞을 지운다).
// 합은 적힌 순서로, 부를 때마다 처음부터 다시 더한다(R14: 들고 다니는 합을 쓰지 않는다).

/**
 * rules.py:99-153. 최근 windowMs 안에 든 시간 구간들의 길이 합(VATS, 분석 불가, 유효 시간 비율에 쓴다).
 *
 * 구간 [start, end)를 시간 순서로, 겹치지 않게 넣는다. 앞 구간의 끝과 바로 이어지면 하나로 합친다.
 * [trim]은 now − windowMs보다 오래된 부분을 정확히 잘라 낸다(걸친 구간은 그 시각에서 자른다).
 * 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class IntervalWindow(private val windowMs: Long) {

    private val starts = ArrayDeque<Long>()
    private val ends = ArrayDeque<Long>()

    /** rules.py:111-113. 원본은 새 목록을 대입한다. 비운 것과 같다. */
    fun clear() {
        starts.clear()
        ends.clear()
    }

    /** rules.py:115-123. 길이가 0 이하인 구간은 넣지 않는다. 앞 구간의 끝에 바로 이어지면 그 끝을 늘린다. */
    fun add(startMs: Long, endMs: Long) {
        if (endMs <= startMs) {
            return
        }
        val n = ends.size
        if (n > 0 && ends[n - 1] == startMs) {
            ends[n - 1] = endMs
        } else {
            starts.addLast(startMs)
            ends.addLast(endMs)
        }
    }

    /** rules.py:125-131. 끝이 now − windowMs 이하인 구간을 지우고, 그 시각에 걸친 첫 구간은 그 시각에서 자른다. */
    fun trim(nowMs: Long) {
        val cutMs = nowMs - windowMs
        while (ends.size > 0 && ends[0] <= cutMs) {
            starts.removeFirst()
            ends.removeFirst()
        }
        if (starts.size > 0 && starts[0] < cutMs) {
            starts[0] = cutMs
        }
    }

    /** rules.py:133-138. 들어 있는 구간 길이의 합. [trim] 뒤에 부르면 최근 windowMs 안의 합이다. */
    fun totalMs(): Long {
        var total = 0L
        for (i in 0 until starts.size) {
            total += ends[i] - starts[i]
        }
        return total
    }

    /** rules.py:140-149. fromMs 이후 부분의 길이 합. */
    fun sinceMs(fromMs: Long): Long {
        var total = 0L
        for (i in 0 until starts.size) {
            var startMs = starts[i]
            if (startMs < fromMs) {
                startMs = fromMs
            }
            if (ends[i] > startMs) {
                total += ends[i] - startMs
            }
        }
        return total
    }

    /**
     * rules.py:151-153. 가장 오래된 구간의 시작. 원본은 비어 있을 때 부르지 않는다(부르면 IndexError).
     * Kotlin은 비어 있으면 NoSuchElementException이다(ArrayDeque.first()).
     */
    fun firstStartMs(): Long = starts.first()
}
