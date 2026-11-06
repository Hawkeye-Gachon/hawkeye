package com.personal.driverguard.data.repository

import android.content.Context
import androidx.core.content.edit
import com.personal.driverguard.domain.repository.ISettingsRepository
import com.personal.driverguard.entity.AnalysisConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json

/** 분석 설정을 SharedPreferences 에 JSON 으로 저장한다. 읽기 실패나 검증 실패 시 기본값을 쓴다. */
class SettingsRepository(context: Context) : ISettingsRepository {

    private val prefs = context.applicationContext.getSharedPreferences("analysis_config", Context.MODE_PRIVATE)
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val _config = MutableStateFlow(load())
    override val config: StateFlow<AnalysisConfig> = _config.asStateFlow()

    override fun update(config: AnalysisConfig): List<String> {
        val errors = config.validate()
        if (errors.isNotEmpty()) return errors
        prefs.edit { putString(KEY, json.encodeToString(AnalysisConfig.serializer(), config)) }
        _config.value = config
        return emptyList()
    }

    override fun resetToDefaults() {
        prefs.edit { remove(KEY) }
        _config.value = AnalysisConfig()
    }

    private fun load(): AnalysisConfig {
        val stored = prefs.getString(KEY, null) ?: return AnalysisConfig()
        return runCatching { json.decodeFromString(AnalysisConfig.serializer(), stored) }
            .getOrNull()
            ?.takeIf { it.validate().isEmpty() }
            ?: AnalysisConfig()
    }

    private companion object {
        const val KEY = "config_json"
    }
}
