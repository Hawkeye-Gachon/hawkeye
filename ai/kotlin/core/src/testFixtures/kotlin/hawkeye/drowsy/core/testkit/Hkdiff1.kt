package hawkeye.drowsy.core.testkit

import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.spec.MiniJson
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * HKDIFF1 파일의 그릇 읽기 (계획 9.2.2절, kotlin/difftools/FORMATS.md 1.1절). 머리와 열을 읽고 검사만 한다.
 * 열의 뜻(세션 자료, 세 모드 재생)은 단계 5가 이 위에 더한다. 시험 고정물이다(배포하지 않는다).
 *
 * 그릇: "HKDIFF1\n", uint32 LE 머리 길이 H, UTF-8 JSON 머리, 8의 배수까지 0 바이트(= 자료 시작 D), 자료 영역.
 * 읽기가 검사하는 것: 표지, 머리 JSON, 채움 바이트가 0, `data_nbytes` = EOF − D, `data_sha256` = 자료 영역의 sha256,
 * 열마다 dtype·shape·nbytes가 맞고 8의 배수 자리에서 시작하며 자료 영역 안에 있다.
 */
class Hkdiff1 private constructor(val header: Map<String, Any?>, private val bytes: ByteArray, private val dataStart: Int) {

    /** 열 하나의 표 (FORMATS.md 1.1절). */
    class Column(val name: String, val dtype: String, val shape: List<Long>, val offset: Long, val nbytes: Long) {
        val count: Long get() = shape.fold(1L) { a, b -> a * b }
    }

    val columns: Map<String, Column>

    init {
        val cols = LinkedHashMap<String, Column>()
        for (c in VectorFile.list(header["columns"], "columns")) {
            val m = VectorFile.obj(c, "column")
            val col = Column(
                name = VectorFile.str(m["name"], "column.name"),
                dtype = VectorFile.str(m["dtype"], "column.dtype"),
                shape = VectorFile.list(m["shape"], "column.shape").map { VectorFile.long(it, "column.shape") },
                offset = VectorFile.long(m["offset"], "column.offset"),
                nbytes = VectorFile.long(m["nbytes"], "column.nbytes"),
            )
            val width = WIDTH[col.dtype] ?: throw IllegalArgumentException("column ${col.name}: unknown dtype ${col.dtype}")
            require(col.count * width == col.nbytes) { "column ${col.name}: nbytes ${col.nbytes} != ${col.count} x $width" }
            require(col.offset % 8 == 0L) { "column ${col.name}: offset ${col.offset} is not 8-aligned" }
            require(col.offset >= 0 && col.offset + col.nbytes <= bytes.size - dataStart) { "column ${col.name}: outside the data region" }
            require(cols.put(col.name, col) == null) { "column ${col.name} is listed twice" }
        }
        columns = cols
    }

    private fun buffer(name: String, dtype: String): Pair<ByteBuffer, Int> {
        val c = columns[name] ?: throw IllegalArgumentException("no column $name")
        require(c.dtype == dtype) { "column $name is ${c.dtype}, not $dtype" }
        val buf = ByteBuffer.wrap(bytes, dataStart + c.offset.toInt(), c.nbytes.toInt()).slice().order(ByteOrder.LITTLE_ENDIAN)
        return buf to c.count.toInt()
    }

    fun f64(name: String): DoubleArray {
        val (b, n) = buffer(name, "f64")
        return DoubleArray(n) { b.getDouble(it * 8) }
    }

    fun f64Bits(name: String): LongArray {
        val (b, n) = buffer(name, "f64")
        return LongArray(n) { b.getLong(it * 8) }
    }

    fun i64(name: String): LongArray {
        val (b, n) = buffer(name, "i64")
        return LongArray(n) { b.getLong(it * 8) }
    }

    fun i32(name: String): IntArray {
        val (b, n) = buffer(name, "i32")
        return IntArray(n) { b.getInt(it * 4) }
    }

    fun u32(name: String): LongArray {
        val (b, n) = buffer(name, "u32")
        return LongArray(n) { b.getInt(it * 4).toLong() and 0xFFFFFFFFL }
    }

    fun u8(name: String): IntArray {
        val (b, n) = buffer(name, "u8")
        return IntArray(n) { b.get(it).toInt() and 0xFF }
    }

    fun f32(name: String): FloatArray {
        val (b, n) = buffer(name, "f32")
        return FloatArray(n) { b.getFloat(it * 4) }
    }

    companion object {
        private val MAGIC = "HKDIFF1\n".toByteArray(Charsets.US_ASCII)
        private val WIDTH = mapOf("u8" to 1L, "i32" to 4L, "u32" to 4L, "i64" to 8L, "f64" to 8L, "f32" to 4L)

        fun read(bytes: ByteArray): Hkdiff1 {
            require(bytes.size >= 12) { "HKDIFF1: file shorter than 12 bytes" }
            for (i in MAGIC.indices) require(bytes[i] == MAGIC[i]) { "HKDIFF1: bad magic" }
            val h = ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
            require(12 + h <= bytes.size) { "HKDIFF1: header length $h runs past the end" }
            val headerEnd = (12 + h).toInt()
            val header = VectorFile.obj(MiniJson.parse(bytes.copyOfRange(12, headerEnd)), "header")
            val dataStart = ((headerEnd + 7) / 8) * 8
            require(dataStart <= bytes.size) { "HKDIFF1: data start $dataStart runs past the end" }
            for (i in headerEnd until dataStart) require(bytes[i] == 0.toByte()) { "HKDIFF1: padding byte $i is not 0" }
            require(header["format"] == "HKDIFF1" && header["format_version"] == 1L) { "HKDIFF1: format or version differs" }
            val nbytes = VectorFile.long(header["data_nbytes"], "data_nbytes")
            require(nbytes == (bytes.size - dataStart).toLong()) { "HKDIFF1: data_nbytes $nbytes != ${bytes.size - dataStart}" }
            val sha = sha256Hex(bytes.copyOfRange(dataStart, bytes.size))
            require(sha == header["data_sha256"]) { "HKDIFF1: data_sha256 ${header["data_sha256"]} != $sha" }
            return Hkdiff1(header, bytes, dataStart)
        }
    }
}
