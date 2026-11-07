package com.personal.driverguard.data.remote

/** 단위 테스트용 메모리 토큰 저장소 */
object FakeTokenStore {
    fun create() = TokenStore(object : TokenPersistence {
        private var pair: Pair<String?, String?> = null to null
        override fun load() = pair
        override fun save(access: String, refresh: String) {
            pair = access to refresh
        }
        override fun clear() {
            pair = null to null
        }
    })
}
