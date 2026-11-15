package hawkeye.drowsy.core

/**
 * 모델 함수 (pipeline.py:65 `ModelFn = Callable[[Sequence[Sequence[float]]], Sequence[float]]`, 계획 4.1절).
 *
 * 120 x 17 원시 스텝 특징(window[0]이 가장 오래된 스텝)을 받아 확률 3개(alert, low_vigilant, drowsy)를 돌려준다.
 * 창의 행 수는 `pipeline.step_features.window_steps`, 열 수는 [N_FEATURES]다. 표준화와 log1p는 모델 그래프 안에서
 * 한다(SPEC 14.2). 돌려준 값이 3개가 아니면 파이프라인이 거부한다(pipeline.py:241-242).
 */
fun interface ModelFn {
    fun predict(window: List<List<Double>>): List<Double>
}
