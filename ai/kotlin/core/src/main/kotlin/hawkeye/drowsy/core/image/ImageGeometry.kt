package hawkeye.drowsy.core.image

import hawkeye.drowsy.core.DrowsyInternalApi

/**
 * 90도 단위 회전과 좌우 반전의 순수 좌표 사상 (계획 3.2절, 5.12절). 새 코드다. 앱 쪽 비트맵 연결이 쓸 수 있다(이 저장소에는
 * 사용처가 없다).
 *
 * 정의(계획 2.4절 G6, G7, 4.2절):
 * - rotationDegrees: 이 프레임을 똑바로 세우려면 **시계 방향**으로 돌려야 하는 각도. 0, 90, 180, 270만 받는다.
 *   90, 270이면 결과 크기는 (h, w)다. Python 도구의 `--rotation`(video_io.rotate_upright)은 반시계 각도라
 *   같은 카메라에서 시계 방향 = (360 − 반시계) mod 360이다.
 * - mirrored: 먼저 세우고, 세운 그림을 좌우로 되돌린다.
 *
 * 픽셀은 행 우선 `IntArray`(index = y * width + x)다. 값을 옮기기만 하고 다시 계산하지 않는다. 상태가 없다(R19).
 */
@DrowsyInternalApi
object ImageGeometry {

    /** 받는 회전 각도인지: 0, 90, 180, 270. */
    @JvmStatic
    fun isValidRotation(rotationDegrees: Int): Boolean =
        rotationDegrees == 0 || rotationDegrees == 90 || rotationDegrees == 180 || rotationDegrees == 270

    /** 결과 그림의 폭. */
    @JvmStatic
    fun outputWidth(width: Int, height: Int, rotationDegrees: Int): Int {
        requireRotation(rotationDegrees)
        return if (rotationDegrees == 90 || rotationDegrees == 270) height else width
    }

    /** 결과 그림의 높이. */
    @JvmStatic
    fun outputHeight(width: Int, height: Int, rotationDegrees: Int): Int {
        requireRotation(rotationDegrees)
        return if (rotationDegrees == 90 || rotationDegrees == 270) width else height
    }

    /**
     * 결과 그림의 자리 (xOut, yOut)의 픽셀이 원래 그림의 어느 자리(행 우선 인덱스)에서 오는지.
     * 반전이 있으면 먼저 결과 자리를 좌우로 되돌린 뒤(세운 그림의 자리), 시계 방향 회전을 거꾸로 따라간다.
     */
    @JvmStatic
    fun sourceIndex(xOut: Int, yOut: Int, width: Int, height: Int, rotationDegrees: Int, mirrored: Boolean): Int {
        val outW = outputWidth(width, height, rotationDegrees)
        val outH = outputHeight(width, height, rotationDegrees)
        require(xOut in 0 until outW && yOut in 0 until outH) { "($xOut, $yOut) is outside the $outW x $outH output" }
        return src(xOut, yOut, width, height, outW, rotationDegrees, mirrored)
    }

    /**
     * 그림 전체를 옮긴 새 배열. 크기는 [outputWidth] x [outputHeight].
     * IllegalArgumentException: 폭이나 높이가 0 이하, 배열 길이가 width * height가 아님, 받지 않는 회전 각도.
     */
    @JvmStatic
    fun transform(pixels: IntArray, width: Int, height: Int, rotationDegrees: Int, mirrored: Boolean): IntArray {
        requireRotation(rotationDegrees)
        require(width > 0 && height > 0) { "image size must be positive, got $width x $height" }
        require(pixels.size.toLong() == width.toLong() * height.toLong()) {
            "pixel array has ${pixels.size} values, expected $width x $height"
        }
        val outW = outputWidth(width, height, rotationDegrees)
        val outH = outputHeight(width, height, rotationDegrees)
        val out = IntArray(pixels.size)
        var k = 0
        for (yOut in 0 until outH) {
            for (xOut in 0 until outW) {
                out[k++] = pixels[src(xOut, yOut, width, height, outW, rotationDegrees, mirrored)]
            }
        }
        return out
    }

    /** 검사 없는 사상. 회전 각도와 자리는 부르는 쪽이 검사했다. */
    private fun src(xOut: Int, yOut: Int, width: Int, height: Int, outW: Int, rotationDegrees: Int, mirrored: Boolean): Int {
        val xr = if (mirrored) outW - 1 - xOut else xOut // 세운 그림의 자리
        val yr = yOut
        val xs: Int
        val ys: Int
        when (rotationDegrees) {
            0 -> { xs = xr; ys = yr }
            90 -> { xs = yr; ys = height - 1 - xr } // 시계 방향 90도: 원래의 왼쪽 아래가 왼쪽 위로 온다
            180 -> { xs = width - 1 - xr; ys = height - 1 - yr }
            else -> { xs = width - 1 - yr; ys = xr } // 270: 원래의 오른쪽 위가 왼쪽 위로 온다
        }
        return ys * width + xs
    }

    private fun requireRotation(rotationDegrees: Int) {
        require(isValidRotation(rotationDegrees)) { "rotationDegrees must be 0, 90, 180 or 270, got $rotationDegrees" }
    }
}
