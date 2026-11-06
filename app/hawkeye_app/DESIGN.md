---
name: DriverGuard
description: Blue-hour drive — an abstract dawn sky whose horizon glow carries the driver state, with Liquid Glass controls floating above it
colors:
  night0: "#0F1C33"
  night1: "#13223D"
  dusk2: "#1B2D4F"
  dusk3: "#24395F"
  horizon: "#3A5A8E"
  haze: "#4A6390"
  sky-white: "#EEF3FB"
  mist: "#A5B7D5"
  mist-dim: "#8598B8"
  ice: "#AFCBFF"
  ice-deep: "#07101F"
  glow-clear: "#5BC8B1"
  glow-dawn: "#F3A65B"
  glow-sunset: "#EC5A5F"
  glow-slate: "#8E9BB6"
  glow-rest: "#9C9AF8"
  lavender: "#D3C4FF"
  error: "#FFA3A3"
typography:
  font: "Pretendard (SIL OFL)"
  display: "38–56sp Bold"
  headline: "22–34sp Bold/SemiBold"
  body: "17sp Regular"
  label: "17sp SemiBold (버튼)"
  footnote: "13sp"
spacing: "4dp 단위, 8dp 셀 (4·8·12·16·24·32·48)"
radius: "8 / 12 / 16 / 22 / 32, 버튼은 캡슐"
control-height: "56dp — 모든 버튼·분할 컨트롤·아이콘 버튼"
---

# DriverGuard 디자인 시스템 — 블루아워 드라이브

## Overview

해 뜨기 직전 고속도로의 하늘을 화면 전체의 바탕으로 쓴다. 지평선의 빛이 운전자 상태를 말하고(시그니처), 조작은 그 위에 뜬 유리(Liquid Glass)다. 다크 고정이다. 차 안(야간·터널·역광)에서 쓰는 장면이 기준이다.

- **콘텐츠층:** 하늘(`SkyBackground`), 불투명 네이비 카드(`InsetGroup`), 카메라 창.
- **조작층:** 상단 캡슐, 조작 레일, 상단 바, 하단 바 — 유리.

이전 버전보다 하늘과 면을 한두 단계 밝혔다(깊은 밤 → 해 뜨기 직전).

## Colors

- **원시 값의 위치:** 원시 값은 `views/theme/Palette.kt`에만 있다. 화면은 `Theme.kt`의 의미 토큰만 참조한다(`colorScheme`, `status`, `sky`, `glass`).
- **글자:** `sky-white`는 night1 위에서 14.2:1, `mist`는 7.8:1, `mist-dim`은 5.4:1이다. dusk2 위에서는 각각 12.3, 6.7, 4.7이다.
- **주요 동작:** `ice` 그라데이션 캡슐에 `ice-deep` 글자. 화면당 하나만 둔다.
- **지평선 빛(상태 전용 예약색):**

  | 상태 | 색 |
  | --- | --- |
  | 관찰 중 | `glow-clear` |
  | 주의 | `glow-dawn` |
  | 휴식 권고 | `glow-sunset` |
  | 분석 불가·준비 | `glow-slate` |
  | 휴식 중 | `glow-rest` |

  앰버와 레드는 '지금 행동' 상태에만 쓴다. 상태는 언제나 문구와 아이콘이 함께 전하고, 색은 그 확인이다.
- **시연 기록:** 예전에 시연으로 저장한 여정은 여정 목록과 리포트에서 `lavender` 배지로 표시한다. 상태색과 겹치지 않는다. 앱에서 시연 모드는 뺐다.

## Typography

- 서체는 Pretendard 하나만 쓴다. 제목은 Bold, 버튼은 17sp SemiBold다.
- 시간은 고정폭 숫자(`TabularNumbers`)로 쓴다.
- 한국어는 어절 단위로 줄을 바꾼다(`LineBreak.WordBreak.Phrase` + `LocaleList("ko")`, Android 13 이상).

## Layout

- **화면 폭 기준(600dp):** 600dp 이상이면 두 영역(가로)으로, 그 미만이면 위에서 아래로 쌓는다.
- **스크롤:** 화면당 하나만 둔다. 유리 바의 높이는 스크롤 안쪽 여백으로 넣고(`PaddingValues.plus`), 스크롤 가장자리는 블러 또는 페이드로 처리한다(`scrollEdge`).
- **주행 화면:** 스크롤하지 않는다.
  - 가로: 상단 캡슐(주행 시간, 운전자 화면 보기), 왼쪽 상태 문구, 오른쪽 아래 조작 레일.
  - 세로: 상단 캡슐, 상태 문구, 조작 레일 순서다. 좁은 화면에서는 상단 캡슐의 '운전자 화면 보기'를 아이콘 버튼으로 줄이고 '경고음만' 안내는 상태 문구 아래로 옮긴다.
- **콘텐츠 최대 폭:** 840dp.

## Liquid Glass

- **쓰는 곳:** 유리는 조작층에만 쓴다. 유리 위에 유리를 겹치지 않는다(`LocalInGlass`). 유리 안의 보조 버튼은 옅은 채움만 쓴다.
- **블러:** Android 12 이상은 Haze 블러(28dp)에 위쪽 광택과 테두리를 더한다. 그 아래 버전은 반투명 채움으로 대신하고, 고대비 텍스트 설정이면 불투명하게 그린다.
- **카메라 영상 위:** 블러와 유리를 올리지 않는다. 상태 문구 뒤의 옅은 그라데이션만 허용한다.

## Components

### Buttons

- **크기:** 모든 버튼은 높이 56dp 캡슐, 좌우 24dp 여백, 17sp SemiBold로 같다. 우선순위는 크기가 아니라 스타일로만 나눈다.
- **종류:**
  - `PrimaryButton`: `ice` 유리.
  - `SecondaryButton`: 무색 유리.
  - `SecondaryIconButton`: 유리 안의 56dp 원.
  - `GlassIconButton`: 떠 있는 56dp 원.

  아이콘 버튼에는 스크린 리더 이름을 반드시 붙인다.
- **누름 상태:** 누르면 96%로 줄고 밝아진다. 애니메이션 끄기 설정에서는 밝기만 바뀐다.

### Segmented Control

- 밝은 썸이 선택한 칸으로 미끄러진다. 라디오 그룹으로 동작하고, 높이는 56dp다.

### Driver Window (카메라 창)

- 주행 화면에서는 기본으로 숨긴다. '운전자 화면 보기'를 누르면 상태 문구 자리에 나타나고, 상태 문구는 영상 안 아래쪽에 들어간다.
- 영상 위에는 얼굴 테두리, 모서리의 개인정보 표시, 아래쪽 상태 문구만 둔다. 블러는 쓰지 않는다. 글자 뒤에만 글자 높이만큼 옅은 그라데이션(70%)을 깔고 글자에 그림자를 주어, 얼굴이 있는 위쪽은 가리지 않는다. 창 테두리는 상태색을 따른다.
- 캘리브레이션 화면에서는 카메라를 고정하고 안내만 스크롤한다.

### Cards, Lists, Dialogs

- **`InsetGroup`:** 불투명 네이비(dusk2 94%) 카드, 반경 22dp. 행은 최소 56dp다.
- **`ConfirmDialog`:** surfaceContainerHigh에 유리 테두리를 두르고, 보조 버튼과 주요 버튼을 같은 크기로 나란히 둔다.
- **`GlassTopBar`:** 뒤로 가기, 제목, 아래 줄(분할 컨트롤) 순서로 놓은 유리 캡슐이다.

## Do's and Don'ts

### Do

- 상태가 바뀌면 지평선 빛, 아이콘, 창 테두리의 색을 함께 바꾸고, 같은 내용을 문구로도 전한다.
- 버튼은 56dp 하나로 맞춘다.
- 새 색은 `Palette.kt`에 추가한 뒤 `Theme.kt`의 의미 토큰으로 연결하고, 대비를 계산한다.

### Don't

- 사진 배경이나 3D 차량·도로 장면을 쓰지 않는다.
- 측정하지 않은 정보(주변 차량, 실제 차선, 휴게소, 도착 시간)를 사실처럼 그리지 않는다.
- 상태색을 경고가 아닌 곳(권한, 설정)에 쓰지 않는다.
- 카메라 영상 위에 블러나 유리를 덮지 않고, 얼굴이 있는 위쪽을 가리지 않는다.
- 라이트 테마로 되돌리지 않는다.
