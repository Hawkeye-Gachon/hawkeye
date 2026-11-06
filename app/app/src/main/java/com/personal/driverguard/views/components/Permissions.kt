package com.personal.driverguard.views.components

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect

enum class PermissionStatus {
    GRANTED,

    /** 아직 요청할 수 있음 */
    REQUESTABLE,

    /** 다시 묻지 않음 상태. 앱 설정에서만 허용할 수 있다. */
    BLOCKED,
}

class PermissionHandle(val status: PermissionStatus, val request: () -> Unit)

/**
 * 권한 상태와 요청 동작. [permissions] 중 하나라도 허용되면 허용으로 본다(위치의 정확/대략처럼).
 * 영구 거부 상태면 request 가 앱 설정 화면을 연다. 설정에서 돌아오면 다시 확인한다.
 */
@Composable
fun rememberPermission(vararg permissions: String): PermissionHandle {
    val context = LocalContext.current
    val activity = LocalActivity.current
    val keys = remember { permissions.toList() }
    var granted by remember { mutableStateOf(context.anyGranted(keys)) }
    var blocked by rememberSaveable { mutableStateOf(false) }

    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        granted = context.anyGranted(keys)
        if (granted) blocked = false
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        granted = context.anyGranted(keys)
        blocked = !granted && activity != null &&
            keys.none { ActivityCompat.shouldShowRequestPermissionRationale(activity, it) }
    }
    val status = when {
        granted -> PermissionStatus.GRANTED
        blocked -> PermissionStatus.BLOCKED
        else -> PermissionStatus.REQUESTABLE
    }
    return PermissionHandle(status) {
        if (status == PermissionStatus.BLOCKED) context.openAppSettings() else launcher.launch(keys.toTypedArray())
    }
}

private fun Context.anyGranted(permissions: List<String>) =
    permissions.any { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }

private fun Context.openAppSettings() {
    startActivity(
        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
    )
}
