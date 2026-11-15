package hawkeye.drowsy.core

import hawkeye.drowsy.core.spec.PipelineSpec
import java.util.Collections

/**
 * MediaPipe 결과의 평범한 값 → RawFrame (계획 4.1절). 원본: drowsy/extract.py:217-246 `result_to_raw_frame`, SPEC 1장 끝.
 * 얼굴 유무는 호출자가 정한다(결과의 랜드마크 목록이 비었는지로만, SPEC 1장). 얼굴이 없으면 [noFace]를 쓴다.
 * 상태가 없다(R19).
 */
object RawFrames {

    /** 얼굴 없는 프레임: lm, bs는 빈 맵, m은 빈 목록 (extract.py:227-228). */
    @JvmStatic
    fun noFace(tMs: Long, w: Int, h: Int): RawFrame =
        RawFrame(tMs = tMs, face = false, w = w, h = h, lm = emptyMap(), bs = emptyMap(), m = emptyList())

    /**
     * 얼굴 0의 값으로 RawFrame을 만든다.
     *
     * - landmarkX/Y: 얼굴 0의 모든 랜드마크 좌표(정규화). lm에는 `spec.landmarks.requiredIndices()`의 44개만 오름차순으로
     *   `Float.toDouble()`로 넣는다(extract.py:240).
     * - blendshapeNames/Scores: 결과의 블렌드셰이프 전부를 이름으로 넣는다(extract.py:241). 같은 이름이 두 번이면 뒤의 값이
     *   남는다(Python dict와 같다).
     * - matrix: 얼굴 변환 행렬 16개, MediaPipe가 준 순서 그대로(열 우선, SPEC 1장 "float[16]을 순서 그대로").
     *   Python은 numpy 4x4를 전치해서 같은 열 우선 순서로 편다(extract.py:242-245).
     *
     * IllegalArgumentException: 필요한 인덱스가 없다(468점 모델, extract.py:234-238), 필요한 블렌드셰이프 이름이 없다
     * (SPEC 1장: 참조 구현은 signals 단계에서 예외를 낸다; 계획 4.1절은 여기서 거부한다), 행렬이 16개가 아니다
     * (extract.py:243-244), x와 y 또는 이름과 점수의 길이가 다르다.
     */
    @JvmStatic
    fun of(
        tMs: Long,
        w: Int,
        h: Int,
        landmarkX: FloatArray,
        landmarkY: FloatArray,
        blendshapeNames: List<String>,
        blendshapeScores: FloatArray,
        matrix: FloatArray,
        spec: PipelineSpec,
    ): RawFrame {
        require(landmarkX.size == landmarkY.size) {
            "landmark x and y arrays differ in length (${landmarkX.size} and ${landmarkY.size})"
        }
        val required = spec.landmarks.requiredIndices()
        val last = required[required.size - 1]
        require(landmarkX.size > last) {
            "FaceLandmarker returned ${landmarkX.size} landmarks but index $last is required " +
                "(the iris points 468..477 come only from the 478-landmark model)"
        }
        val lm = LinkedHashMap<Int, Point>(required.size * 2)
        for (index in required) {
            lm[index] = Point(landmarkX[index].toDouble(), landmarkY[index].toDouble())
        }

        require(blendshapeNames.size == blendshapeScores.size) {
            "blendshape names and scores differ in length (${blendshapeNames.size} and ${blendshapeScores.size})"
        }
        val bs = LinkedHashMap<String, Double>(blendshapeNames.size * 2)
        for (i in blendshapeNames.indices) {
            bs[blendshapeNames[i]] = blendshapeScores[i].toDouble()
        }
        for (name in spec.blendshapes.requiredNames()) {
            require(bs.containsKey(name)) { "FaceLandmarker result has no blendshape named '$name'" }
        }

        require(matrix.size == 16) { "transformation matrix must have 16 values, got ${matrix.size}" }
        val m = ArrayList<Double>(16)
        for (v in matrix) m.add(v.toDouble())

        return RawFrame(
            tMs = tMs,
            face = true,
            w = w,
            h = h,
            lm = Collections.unmodifiableMap(lm),
            bs = Collections.unmodifiableMap(bs),
            m = Collections.unmodifiableList(m),
        )
    }
}
