package com.personal.driverguard.ui.state

import android.content.Context
import com.personal.driverguard.data.remote.EndDriveRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class PendingUploadsTest {
    private val prefs = RuntimeEnvironment.getApplication().getSharedPreferences("pending_test", Context.MODE_PRIVATE)
    private val store = PendingUploads(prefs)

    private fun item(id: String, owner: String?, distance: Int = 0) =
        PendingUpload(id, "2026-11-07T10:00:00+09:00", null, EndDriveRequest("2026-11-07T10:30:00+09:00", distance, emptyList(), emptyList(), emptyList()), owner)

    @Before
    fun clear() {
        prefs.edit().clear().commit()
    }

    @Test
    fun putReplacesTheSameDriveAndRemoveDeletesOnlyThatDrive() {
        store.put(item("a", null, distance = 100))
        store.put(item("b", null))
        store.put(item("a", null, distance = 250))
        assertEquals(listOf("b", "a"), store.all().map { it.clientDriveId })
        assertEquals("중간 저장은 같은 주행을 덮어쓴다", 250, store.all().first { it.clientDriveId == "a" }.body.distanceMeters)
        store.remove("a")
        assertEquals(listOf("b"), store.all().map { it.clientDriveId })
    }

    @Test
    fun onlyTheOwnersRecordsAreUploadable() {
        store.put(item("mine", "me@example.com"))
        store.put(item("theirs", "other@example.com"))
        store.put(item("legacy", null))
        store.owner = " Me@Example.com "
        assertEquals("계정은 대소문자·공백을 무시하고 맞춘다", listOf("mine", "legacy"), store.uploadable().map { it.clientDriveId })
        assertEquals("진행 중인 주행은 뺀다", listOf("legacy"), store.uploadable(except = "mine").map { it.clientDriveId })
        store.owner = null
        assertNull(store.owner)
        assertEquals("로그인 계정을 모르면 계정이 적히지 않은 기록만 올린다", listOf("legacy"), store.uploadable().map { it.clientDriveId })
        assertEquals("다른 계정의 기록은 지우지 않고 남겨 둔다", 3, store.all().size)
    }

    @Test
    fun recordsSavedBeforeOwnerExistedStillLoad() {
        prefs.edit().putString(
            "pending",
            """[{"clientDriveId":"old","startedAt":"2026-11-07T10:00:00+09:00","body":{"endedAt":"2026-11-07T10:30:00+09:00","distanceMeters":0,"events":[],"route":[],"waypoints":[]}}]""",
        ).commit()
        val loaded = store.all().single()
        assertEquals("old", loaded.clientDriveId)
        assertNull(loaded.owner)
        assertTrue(store.uploadable().isNotEmpty())
    }

    @Test
    fun brokenStorageIsTreatedAsEmpty() {
        prefs.edit().putString("pending", "{not json").commit()
        assertTrue(store.all().isEmpty())
        store.put(item("a", null))
        assertEquals(1, store.all().size)
    }
}
