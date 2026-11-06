package com.personal.driverguard.domain.repository

import com.personal.driverguard.entity.AnalysisConfig
import kotlinx.coroutines.flow.StateFlow

interface ISettingsRepository {
    val config: StateFlow<AnalysisConfig>

    /** 검증에 실패하면 저장하지 않고 오류 목록을 돌려준다. */
    fun update(config: AnalysisConfig): List<String>

    fun resetToDefaults()
}
