package com.personal.driverguard.ui.state

import android.content.SharedPreferences
import com.personal.driverguard.data.remote.HawkeyeApi
import kotlinx.serialization.builtins.ListSerializer

/**
 * 아직 서버에 올리지 못한 주행 기록 보관함(기기 안, SharedPreferences).
 *
 * 두 가지가 들어온다.
 *  - 종료 업로드에 실패한 주행
 *  - 진행 중인 주행의 중간 저장본(앱이 강제 종료돼도 그때까지의 기록이 남도록 주기적으로 덮어쓴다)
 *
 * 기록에는 로그인한 계정([owner])을 붙여 둔다. 다른 계정으로 로그인한 뒤 다시 올릴 때 남의 기록이 섞이지 않게 하기 위해서다.
 * 여러 스레드에서 불러도 된다.
 */
class PendingUploads(private val prefs: SharedPreferences) {
    private val serializer = ListSerializer(PendingUpload.serializer())

    /** 지금 로그인한 계정(로그인 이메일을 소문자로). 로그아웃하면 null. 앱을 다시 켜도 유지된다. */
    var owner: String?
        get() = prefs.getString(KEY_OWNER, null)
        set(value) {
            prefs.edit().apply { if (value == null) remove(KEY_OWNER) else putString(KEY_OWNER, normalize(value)) }.apply()
        }

    @Synchronized
    fun all(): List<PendingUpload> = prefs.getString(KEY_PENDING, null)
        ?.let { runCatching { HawkeyeApi.json.decodeFromString(serializer, it) }.getOrNull() }
        .orEmpty()

    /** 같은 주행([PendingUpload.clientDriveId])이 있으면 덮어쓴다 */
    @Synchronized
    fun put(item: PendingUpload) = write(all().filterNot { it.clientDriveId == item.clientDriveId } + item)

    @Synchronized
    fun remove(clientDriveId: String) {
        val list = all()
        if (list.any { it.clientDriveId == clientDriveId }) write(list.filterNot { it.clientDriveId == clientDriveId })
    }

    /**
     * 지금 계정이 올려도 되는 기록. 계정이 적히지 않은 예전 기록은 누구든 올린다(이전 버전과 같은 동작).
     * [except] 는 진행 중인 주행으로, 종료 전에 올리면 안 되므로 뺀다.
     */
    fun uploadable(except: String? = null): List<PendingUpload> {
        val me = owner
        return all().filter { it.clientDriveId != except && (it.owner == null || it.owner == me) }
    }

    private fun write(list: List<PendingUpload>) {
        prefs.edit().putString(KEY_PENDING, HawkeyeApi.json.encodeToString(serializer, list)).apply()
    }

    companion object {
        private const val KEY_PENDING = "pending"
        private const val KEY_OWNER = "owner"

        fun normalize(email: String) = email.trim().lowercase()
    }
}
