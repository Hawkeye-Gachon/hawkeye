-- DriverGuard schema (PostgreSQL)

-- UPDATE 시 updated_at 자동 갱신
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =========================================================
-- users
-- =========================================================
CREATE TABLE users (
    user_id          BIGINT       GENERATED ALWAYS AS IDENTITY,
    email            VARCHAR(254),
    name             VARCHAR(50)  NOT NULL,
    status           VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    age_confirmed_at TIMESTAMPTZ  NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    withdrawn_at     TIMESTAMPTZ,
    CONSTRAINT pk_users PRIMARY KEY (user_id),
    CONSTRAINT ck_users_status CHECK (status IN ('ACTIVE', 'WITHDRAWN')),
    CONSTRAINT ck_users_email_lower CHECK (email = lower(email))
);

-- ACTIVE 사용자 사이에서만 이메일 UNIQUE (탈퇴 후 같은 이메일로 재가입 가능)
CREATE UNIQUE INDEX uq_users_email_active ON users (email) WHERE status = 'ACTIVE';

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- auths
-- =========================================================
CREATE TABLE auths (
    auth_id          BIGINT       GENERATED ALWAYS AS IDENTITY,
    user_id          BIGINT       NOT NULL,
    provider         VARCHAR(20)  NOT NULL,
    provider_user_id VARCHAR(255),
    password_hash    VARCHAR(255),
    last_login_at    TIMESTAMPTZ,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_auths PRIMARY KEY (auth_id),
    CONSTRAINT fk_auths_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    CONSTRAINT uq_auths_user_provider UNIQUE (user_id, provider),
    CONSTRAINT uq_auths_provider_user UNIQUE (provider, provider_user_id),
    -- EMAIL 이면 password_hash 만, 소셜이면 provider_user_id 만 값이 있어야 함
    CONSTRAINT ck_auths_credential CHECK (
        (provider = 'EMAIL' AND password_hash IS NOT NULL AND provider_user_id IS NULL)
        OR
        (provider <> 'EMAIL' AND provider_user_id IS NOT NULL AND password_hash IS NULL)
    )
);

CREATE TRIGGER trg_auths_updated_at
    BEFORE UPDATE ON auths
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- refresh_tokens : 로그인(기기) 1건당 1행, 토큰 원문이 아닌 SHA-256 해시만 저장
-- =========================================================
CREATE TABLE refresh_tokens (
    token_id    BIGINT       GENERATED ALWAYS AS IDENTITY,
    user_id     BIGINT       NOT NULL,
    token_hash  VARCHAR(64)  NOT NULL,
    expires_at  TIMESTAMPTZ  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_refresh_tokens PRIMARY KEY (token_id),
    CONSTRAINT fk_refresh_tokens_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    CONSTRAINT uq_refresh_tokens_hash UNIQUE (token_hash)
);

CREATE INDEX ix_refresh_tokens_user ON refresh_tokens (user_id);

-- =========================================================
-- rest_places : 졸음쉼터 + 고속도로 휴게소 (place_type 으로 구분)
--   주 1회 수집(RestPlaceSyncService)으로 upsert, 이번 수집에 없던 곳은 active = FALSE
-- =========================================================
CREATE TABLE rest_places (
    place_id            BIGINT            GENERATED ALWAYS AS IDENTITY,
    place_type          VARCHAR(20)       NOT NULL,
    source_key          VARCHAR(200)      NOT NULL,
    name                VARCHAR(100)      NOT NULL,
    route_name          VARCHAR(50),
    direction           VARCHAR(100),
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    address             VARCHAR(255),
    parking_spaces      INTEGER,
    active              BOOLEAN           NOT NULL DEFAULT TRUE,
    synced_at           TIMESTAMPTZ       NOT NULL DEFAULT now(),
    created_at          TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_rest_places PRIMARY KEY (place_id),
    CONSTRAINT uq_rest_places_source UNIQUE (place_type, source_key),
    CONSTRAINT ck_rest_places_type CHECK (place_type IN ('REST_AREA', 'DROWSY_SHELTER')),
    -- 좌표는 둘 다 없거나, 둘 다 국내 범위 안
    CONSTRAINT ck_rest_places_coord CHECK (
        (latitude IS NULL AND longitude IS NULL)
        OR (latitude BETWEEN 33 AND 39 AND longitude BETWEEN 124 AND 132)
    )
);

CREATE TRIGGER trg_rest_places_updated_at
    BEFORE UPDATE ON rest_places
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- terms : 약관 버전별 1행. 개정하면 새 version 행 추가
--   code 별로 effective_at <= now() 인 것 중 가장 높은 version 이 "현재 약관"
-- =========================================================
CREATE TABLE terms (
    terms_id       BIGINT        GENERATED ALWAYS AS IDENTITY,
    code           VARCHAR(30)   NOT NULL,
    version        INTEGER       NOT NULL,
    title          VARCHAR(100)  NOT NULL,
    content        TEXT          NOT NULL,
    is_required    BOOLEAN       NOT NULL,
    display_order  INTEGER       NOT NULL,
    effective_at   TIMESTAMPTZ   NOT NULL,
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT pk_terms PRIMARY KEY (terms_id),
    CONSTRAINT uq_terms_code_version UNIQUE (code, version),
    CONSTRAINT ck_terms_version CHECK (version > 0)
);

-- =========================================================
-- user_terms_agreements : 사용자 × 약관 버전별 동의 기록
-- =========================================================
CREATE TABLE user_terms_agreements (
    agreement_id  BIGINT       GENERATED ALWAYS AS IDENTITY,
    user_id       BIGINT       NOT NULL,
    terms_id      BIGINT       NOT NULL,
    agreed        BOOLEAN      NOT NULL,
    agreed_at     TIMESTAMPTZ,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_user_terms_agreements PRIMARY KEY (agreement_id),
    CONSTRAINT fk_user_terms_agreements_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    CONSTRAINT fk_user_terms_agreements_terms FOREIGN KEY (terms_id) REFERENCES terms (terms_id),
    CONSTRAINT uq_user_terms_agreements_user_terms UNIQUE (user_id, terms_id),
    -- 동의했으면 동의 시각 필수, 미동의면 NULL
    CONSTRAINT ck_user_terms_agreements_agreed_at CHECK (
        (agreed AND agreed_at IS NOT NULL) OR (NOT agreed AND agreed_at IS NULL)
    )
);

CREATE TRIGGER trg_user_terms_agreements_updated_at
    BEFORE UPDATE ON user_terms_agreements
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- drives : 관찰 1회. 종료 업로드 때 계산한 요약값을 함께 저장
--   IN_PROGRESS 관찰 중 / ENDED 종료 업로드 완료 / ABANDONED 종료 업로드 없이 다음 주행이 시작됨
-- =========================================================
CREATE TABLE drives (
    drive_id           BIGINT       GENERATED ALWAYS AS IDENTITY,
    user_id            BIGINT       NOT NULL,
    client_drive_id    UUID         NOT NULL,
    status             VARCHAR(20)  NOT NULL DEFAULT 'IN_PROGRESS',
    started_at         TIMESTAMPTZ  NOT NULL,
    ended_at           TIMESTAMPTZ,
    observed_seconds   INTEGER,
    distance_meters    INTEGER,
    inattention_count  INTEGER      NOT NULL DEFAULT 0,
    drowsiness_count   INTEGER      NOT NULL DEFAULT 0,
    route_simplified   JSONB,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT pk_drives PRIMARY KEY (drive_id),
    CONSTRAINT fk_drives_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    CONSTRAINT uq_drives_user_client UNIQUE (user_id, client_drive_id),
    CONSTRAINT ck_drives_status CHECK (status IN ('IN_PROGRESS', 'ENDED', 'ABANDONED')),
    -- 종료 업로드가 끝난 주행만 종료 시각이 있음
    CONSTRAINT ck_drives_ended_at CHECK (
        (status = 'ENDED' AND ended_at IS NOT NULL AND ended_at >= started_at)
        OR (status <> 'ENDED' AND ended_at IS NULL)
    ),
    CONSTRAINT ck_drives_counts CHECK (
        inattention_count >= 0 AND drowsiness_count >= 0
        AND (observed_seconds IS NULL OR observed_seconds >= 0)
        AND (distance_meters IS NULL OR distance_meters >= 0)
    )
);

-- 사용자당 진행 중인 주행은 1개
CREATE UNIQUE INDEX uq_drives_user_in_progress ON drives (user_id) WHERE status = 'IN_PROGRESS';

-- 주행 목록 · 가장 최근 주행일 (종료 업로드가 끝난 주행만)
CREATE INDEX ix_drives_user_ended_started ON drives (user_id, started_at DESC) WHERE status = 'ENDED';

CREATE TRIGGER trg_drives_updated_at
    BEFORE UPDATE ON drives
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- drive_events : 부주의·졸음 감지 1건
-- =========================================================
CREATE TABLE drive_events (
    event_id     BIGINT            GENERATED ALWAYS AS IDENTITY,
    drive_id     BIGINT            NOT NULL,
    event_type   VARCHAR(20)       NOT NULL,
    detected_at  TIMESTAMPTZ       NOT NULL,
    latitude     DOUBLE PRECISION,
    longitude    DOUBLE PRECISION,
    created_at   TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_drive_events PRIMARY KEY (event_id),
    CONSTRAINT fk_drive_events_drive FOREIGN KEY (drive_id) REFERENCES drives (drive_id) ON DELETE CASCADE,
    -- 같은 종류·같은 시각 중복 방지 + 주행별 시각순 조회 인덱스 역할
    CONSTRAINT uq_drive_events_drive_type_detected UNIQUE (drive_id, event_type, detected_at),
    CONSTRAINT ck_drive_events_type CHECK (event_type IN ('INATTENTION', 'DROWSINESS')),
    CONSTRAINT ck_drive_events_coord CHECK (
        (latitude IS NULL AND longitude IS NULL)
        OR (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)
    )
);

-- =========================================================
-- drive_waypoints : 쉼터 경유지 추가 기록 (장소 정보는 저장 시점 스냅샷)
--   방문 여부는 앱이 기록해서 보냄 (visited_at)
-- =========================================================
CREATE TABLE drive_waypoints (
    waypoint_id   BIGINT            GENERATED ALWAYS AS IDENTITY,
    drive_id      BIGINT            NOT NULL,
    place_id      BIGINT            NOT NULL,
    place_name    VARCHAR(100)      NOT NULL,
    place_type    VARCHAR(20)       NOT NULL,
    latitude      DOUBLE PRECISION  NOT NULL,
    longitude     DOUBLE PRECISION  NOT NULL,
    nav_mode      VARCHAR(20)       NOT NULL,
    trigger_type  VARCHAR(30)       NOT NULL,
    added_at      TIMESTAMPTZ       NOT NULL,
    visited       BOOLEAN           NOT NULL,
    visited_at    TIMESTAMPTZ,
    created_at    TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_drive_waypoints PRIMARY KEY (waypoint_id),
    CONSTRAINT fk_drive_waypoints_drive FOREIGN KEY (drive_id) REFERENCES drives (drive_id) ON DELETE CASCADE,
    CONSTRAINT fk_drive_waypoints_place FOREIGN KEY (place_id) REFERENCES rest_places (place_id),
    CONSTRAINT ck_drive_waypoints_place_type CHECK (place_type IN ('REST_AREA', 'DROWSY_SHELTER')),
    CONSTRAINT ck_drive_waypoints_nav_mode CHECK (nav_mode IN ('WAYPOINT', 'DESTINATION')),
    CONSTRAINT ck_drive_waypoints_trigger CHECK (trigger_type IN ('DROWSINESS', 'INATTENTION_REPEATED')),
    -- 방문이면 방문 시각 필수, 아니면 NULL
    CONSTRAINT ck_drive_waypoints_visited CHECK (
        (visited AND visited_at IS NOT NULL) OR (NOT visited AND visited_at IS NULL)
    )
);

CREATE INDEX ix_drive_waypoints_drive ON drive_waypoints (drive_id);

-- =========================================================
-- drive_hourly_stats : 주행 × 서울 기준 1시간 구간 집계 (리포트 전용)
--   종료 업로드 때 한 번 계산해 저장. 거리는 구간별 관찰 시간 비율로 나눔
-- =========================================================
CREATE TABLE drive_hourly_stats (
    stat_id            BIGINT       GENERATED ALWAYS AS IDENTITY,
    drive_id           BIGINT       NOT NULL,
    user_id            BIGINT       NOT NULL,
    hour_start         TIMESTAMPTZ  NOT NULL,
    observed_seconds   INTEGER      NOT NULL,
    inattention_count  INTEGER      NOT NULL DEFAULT 0,
    drowsiness_count   INTEGER      NOT NULL DEFAULT 0,
    distance_meters    INTEGER      NOT NULL DEFAULT 0,
    CONSTRAINT pk_drive_hourly_stats PRIMARY KEY (stat_id),
    CONSTRAINT fk_drive_hourly_stats_drive FOREIGN KEY (drive_id) REFERENCES drives (drive_id) ON DELETE CASCADE,
    CONSTRAINT fk_drive_hourly_stats_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    CONSTRAINT uq_drive_hourly_stats_drive_hour UNIQUE (drive_id, hour_start),
    CONSTRAINT ck_drive_hourly_stats_values CHECK (
        observed_seconds BETWEEN 0 AND 3600
        AND inattention_count >= 0 AND drowsiness_count >= 0 AND distance_meters >= 0
    )
);

CREATE INDEX ix_drive_hourly_stats_user_hour ON drive_hourly_stats (user_id, hour_start);
