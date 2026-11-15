"""실시간 시험: 카메라 스트림에 전체 파이프라인을 돌려 상태를 터미널과 웹 페이지에 바로 보여 준다. 이식 대상이 아니다.

데스크톱 시험 도구다. 저장소는 WSL2 안에 있고 카메라는 Windows에 붙어 있어서(WSL에는 `/dev/video*`가 없다),
Windows의 ffmpeg가 카메라를 MPEG-TS(H.264, 지연을 줄인 설정)로 TCP `localhost`에 보내고 이 프로그램이 받아서 푼다.

흐름
    소스 → 읽기 스레드(`LatestFrameReader`: PyAV 디코딩, 가장 새 프레임만 넘김) → 처리 루프(`LiveLoop`)의
    `LiveSession`(`drowsy.extract`의 VIDEO 모드 FaceLandmarker → `DriverPipeline`) → 터미널 줄 + 웹 페이지(`StatusServer`)
    세션이 끝나면: 프레임 표(parquet) → `python -m drowsy.demo --raw`와 같은 타임라인 PNG와 이벤트 JSON
- 신호, closure, 감김 검출, 스텝 특징, 규칙, 경고는 여기서 계산하지 않는다. `DriverPipeline`이 낸 값을 그대로 보여 준다.
  파이프라인은 데모와 같은 인자로 만든다(`DriverPipeline(spec, rules, None, None, model)`): 기준선을 주지 않으므로
  세션 자신의 앞 60초에서 보정한다. 경고의 종류(`immediate`, `deferred`, `repeat`)도 데모의 `AlertKinds`로 가린다.
- 모델을 쓰기 전에 데모와 같은 대조를 한다(`demo.verify_model`, `demo.make_model_fn`). 어긋나면 실행하지 않는다.
- 얼굴 화소는 디스크에 쓰지 않는다(PLAN 1.2절의 결정 D13). 디코딩한 영상은 메모리에서 MediaPipe와 미리보기 JPEG로만
  간다. 출력 폴더에는 프레임 표(랜드마크 값만), PNG, JSON 세 파일만 생기고, 이 프로그램은 그 밖에 아무것도 쓰지 않는다.
  다만 모델을 돌리는 onnxruntime 라이브러리가 자기 파일을 따로 둔다: 임시 폴더의 작은 파일 하나와 사용자 캐시 폴더
  아래의 작은 데이터베이스다. 둘 다 화소를 담지 않는다(`import onnxruntime`만으로도 생긴다. `python -m drowsy.demo`도 같다).

소스 (`--source`. PyAV가 열 수 있는 것)
    `tcp://127.0.0.1:5599?listen=1`  기본값. 이 프로그램이 듣고 보내는 쪽이 접속한다. MPEG-TS로 읽는다.
    `/dev/video0` 같은 장치          v4l2로 연다. 이 PC의 WSL에는 카메라 장치가 없어 실행해 보지 못했다.
    영상 파일                        시험용. 파일의 타임스탬프대로 실제 속도로 재생한다(아래 가정).
  보내는 쪽의 예(Windows PowerShell의 ffmpeg. 카메라 입력 부분은 실행해 보지 못했다. 아래 가정):
    ffmpeg -hide_banner -list_devices true -f dshow -i dummy                     카메라 이름을 본다
    ffmpeg -hide_banner -list_options true -f dshow -i video="<카메라 이름>"      그 카메라의 해상도와 프레임률을 본다
    ffmpeg -hide_banner -f dshow -video_size 640x480 -framerate 30 -i video="<카메라 이름>" -c:v libx264
           -preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p -f mpegts tcp://localhost:5599
  - 이 프로그램을 먼저 실행해 듣게 한 뒤 보내는 쪽을 시작한다. 먼저 시작한 보내는 쪽은 접속이 거부되어 끝난다. 셋째
    명령은 시작할 때 터미널에 실제 포트로 나오고(`SENDER_COMMAND`), 셋 모두 `--help`에 있다.
  - `-g 30`은 키프레임 간격이다(30 fps에서 1초). 세션은 첫 키프레임에서 시작하므로 짧게 둔다. 키프레임 옵션이 없는
    보내는 쪽(libx264의 기본 간격)에서는 첫 프레임이 보내는 쪽을 시작한 지 7.92초 뒤에 왔다(2026-11-14, 검토 때 이 PC의
    Linux ffmpeg 7.0.2와 합성 그림으로 잰 값. `-g 30`이면 0.59초).
  - `-re`와 출력 쪽 `-r`, `-fps_mode cfr`은 붙이지 않는다. 같은 검토에서 `-fps_mode cfr`은 프레임 시각을 고른 간격으로
    바꾸고 빈 구간을 복제 프레임으로 채웠다. 파이프라인의 시간은 프레임 시각에서 나온다.
  - 이 프로그램이 기다리는 동안 보내는 쪽의 접속이 거부되거나, 보내는 쪽을 시작했는데도 이 프로그램이 `still waiting`
    줄만 내면 주소를 `tcp://127.0.0.1:5599`로 바꿔 본다(아래 가정).
  - 이 프로그램을 먼저 끝내면 보내는 쪽은 스스로 끝난다(같은 검토에서 ffmpeg는 "Broken pipe"를 알리고 끝났다).

시각 (`iter_live_frames`. 규약은 `drowsy.video_io`와 같다)
- `t_ms = 1 + floor((pts − 첫 프레임 pts)·time_base·1000 + 0.5)` (`video_io.pts_to_ms`). 첫 프레임이 1이고 엄격히 증가한다.
- `t_ms`가 이전 값보다 크지 않은 프레임, pts가 있는 스트림에서 pts가 없는 프레임은 MediaPipe에 넣기 전에 버리고
  센다(`StreamCounts.skipped`. VIDEO 모드는 엄격히 증가하는 값만 받는다).
- 첫 프레임에 pts가 없는 스트림은 도착 시각(단조 시계)으로 시각을 만든다(`StreamCounts.arrival_clock`).
- 회전: `--rotation`(반시계, 기본 0)만큼 돌려 세운다. 실시간 스트림에는 회전 자동 점검이 없다.

설계 결정 (계획서가 구현에 맡긴 것)
1. 읽기 스레드와 처리 루프 사이의 대기열은 칸이 하나다(`LatestFrameReader`). "가장 새 프레임"은 읽기 스레드가 시각을
   받아들인 프레임 가운데 처리 루프가 아직 가져가지 않은 마지막 것이다. 칸에 프레임이 남아 있는데 다음 프레임이 오면
   남아 있던 것을 버리고 센다(`dropped`). 그래서 처리가 밀려도 밀린 프레임이 쌓이지 않고(다음에 처리하는 것은 언제나
   가장 새 프레임이다), 버린 프레임은 파이프라인에 프레임 간격으로 나타난다(앱의 프레임 누락과 같다). 평균으로는
   따라갈 수 있어도 프레임이 몰려서 오면 버린다(스트림이 끝날 때 디코더가 한꺼번에 내놓는 마지막 프레임들이 그럴 수 있다).
2. 상태는 불변 조각으로 HTTP 스레드에 넘긴다(`StatusServer`). 처리 루프가 JSON 바이트와 JPEG 바이트를 만들어 잠금
   안에서 참조만 바꾸고, HTTP 스레드는 잠금 안에서 그 참조를 읽는다. 바이트는 만든 뒤에 고치지 않는다.
3. 미리보기는 처리 루프가 방금 처리한 프레임으로 `PREVIEW_INTERVAL_S`(0.1초)에 한 번까지 만든다. 긴 변이
   `PREVIEW_MAX_SIDE_PX`(480)를 넘으면 줄이고 JPEG 품질은 `PREVIEW_JPEG_QUALITY`(70)다. MediaPipe에 넘긴 그대로(세운 뒤,
   좌우를 뒤집지 않은) 그림이다. 메모리에만 있고 127.0.0.1에서만 내주며 `--no-preview`면 만들지도 않는다.
4. 세션이 끝나는 길
   - Ctrl+C(SIGINT), kill(SIGTERM), 터미널 창을 닫음(SIGHUP): 처리기가 표시만 세우고 루프가 처리 중인 프레임을 마친 뒤
     끝낸다(프레임 하나가 반만 처리되는 일이 없다). 표시가 선 뒤에는 새 프레임을 시작하지 않는다(기다리던 프레임이 그
     뒤에 도착해도 처리하지 않는다). 파일을 쓰는 동안의 신호는 무시한다. 명령으로 실행했으면(`python -m`) 파일을 다 쓴
     뒤에도 Ctrl+C를 무시 상태로 둔다(여러 번 눌러도 종료 코드가 바뀌지 않는다).
   - 터미널 창을 닫은 뒤에는 터미널에 줄을 낼 수 없다. 줄은 버리고 파일은 쓴다. 그때의 종료 코드는 뜻이 없다(가짜
     터미널을 닫아 본 2026-11-14의 실행에서는 120이었다: 인터프리터가 끝날 때 표준 출력을 비우지 못한다).
   - 스트림의 끝: 보내는 쪽이 멈추면 칸에 남은 프레임까지 처리하고 끝낸다.
   - 멈춘 스트림: `STALL_TIMEOUT_S`(10초) 동안 데이터가 없으면 끝낸다(`clock.reset_gap_ms`와 같은 길이: 그보다 긴
     끊김 뒤에는 어차피 검출기와 누적기가 초기화된다).
   - 프레임 상한: 아래 5.
   위의 길은 프레임을 하나라도 처리했으면 세 파일을 쓰고 0으로 끝난다.
   - 첫 프레임을 처리하기 전에 멈추라는 요청이 오면 아무것도 쓰지 않고 2로 끝난다. 보내는 쪽을 기다리는 중의 Ctrl+C는
     바로 끝난다(그때의 SIGTERM과 SIGHUP은 처리기를 달기 전이라 프로세스를 그대로 끝낸다: 잃을 세션이 없다). 그래서
     보내는 쪽을 기다리는 동안 kill(SIGTERM)이나 터미널 창을 닫아(SIGHUP) 끝내면, 시작할 때 만든 출력 폴더가 빈 채로
     남는다(아래 "쓸 수 없는 출력 폴더" 항목의 예외다).
   - 멈추라는 요청부터 프로세스가 끝날 때까지: 스트림이 오는 동안에는 0.6초에서 0.8초쯤(세 파일을 쓰는 시간 포함), 보내는
     쪽이 접속한 채 아무것도 보내지 않는 동안에는 3초쯤이다(읽기 스레드가 읽기에 막혀 있어서 `STOP_JOIN_S`(2초)를 한 번
     기다린다). 2026-11-14에 이 PC에서 여러 번 잰 값이다: 스트림이 오는 동안 0.60초부터 0.77초(프로그램을 가장 낮은
     우선순위(`nice -n 19`)로 돌린 실행에서는 0.87초), 접속한 채 조용한 동안 2.72초와 2.92초. 접속된 뒤 첫 키프레임을
     기다리는 동안에는 한 번 재어 0.51초였다(읽기 스레드가 패킷마다 표시를 본다).
   - 보내는 쪽이 접속하지 않음: `CONNECT_TIMEOUT_S`(120초)를 기다린 뒤 오류 문장과 함께 2로 끝난다. 기다리는 동안
     `WAIT_NOTE_INTERVAL_S`(30초)마다 남은 시간을 알린다(남은 시간이 0초로 적힐 알림은 내지 않는다). 접속한 채 MPEG-TS
     영상을 보내지 않는 쪽도 같은 길로 끝나므로, 그 오류 문장은 두 경우(아무도 접속하지 않았다, 접속한 쪽이 MPEG-TS
     영상을 보내지 않았다)를 함께 적는다. 접속한 쪽이 영상 없이 끊으면(포트만 확인해 본 접속, MPEG-TS가 아닌 것을
     보내고 끊은 쪽) 그 사실을 알리고 남은 시간 동안 다시 듣는다.
   - 쓸 수 없는 출력 폴더: 시작할 때 폴더를 만들고 그 안에 빈 파일 하나를 만들었다 지워 확인한다. 안 되면 아무것도
     시작하지 않고 2로 끝난다(표는 세션이 끝날 때 한 번 쓰므로, 끝에 가서 알면 세션을 잃는다). 아무것도 쓰지 않고 끝나는
     실행은 새로 만든 폴더를 되돌린다. 예외: 보내는 쪽을 기다리는 동안 SIGTERM이나 터미널 창을 닫아 끝난 프로세스는
     되돌리지 못한다(빈 폴더가 남는다. 위 참조).
   - 웹 페이지의 포트가 사용 중: 소스를 열기 전에 오류 문장(`--port`를 알려 준다)과 함께 2로 끝난다.
   - 세션 도중 처리 오류(MediaPipe, 파이프라인. 오류의 종류를 가리지 않는다): 그때까지의 프레임으로 세 파일을 쓰고 오류
     문장과 함께 2로 끝난다. 미리보기를 만들다 난 오류는 미리보기만 끄고, 웹 페이지의 상태를 내다 난 오류는 한 번 알리고
     세션을 계속한다(둘은 표시일 뿐이다).
   - 프로그램이 뜨는 동안(라이브러리를 올리는 처음 2초쯤. 검토 때 잰 값 1.7초)의 Ctrl+C는 Python의 KeyboardInterrupt
     traceback으로 끝난다(아직 아무것도 시작하지 않았다).
5. 프레임 표는 `extract.write_raw_frames`의 열 버퍼에만 쌓인다(프레임마다 Python 객체를 들고 있지 않는다).
   `write_raw_frames`가 프레임을 하나 달라고 할 때마다 처리 루프가 프레임 하나를 처리해 넘기는 방식이다(데모가
   `run_pipeline`에 프레임 생성기를 넘기는 것과 같다). 한 세션의 상한은 `MAX_SESSION_FRAMES`(216,000 프레임, 30 fps로
   2시간)이고, 상한에 이르면 세션을 끝내고 파일을 쓴다. 상한까지 찬 표(얼굴 프레임 216,000개)는 열 버퍼가 276 MB
   (프레임당 1,339바이트)였고, parquet을 쓰는 동안의 최대 상주 메모리는 시작 전보다 839 MB 많았다(2026-11-14, 이 PC).
   `LiveSession`과 처리 루프는 프레임마다 쌓는 것이 없다: 웹 페이지에 남기는 최근 이벤트와 경고는 `RECENT_ITEMS`(20)개
   까지이고, 이벤트 전체는 세션이 끝난 뒤 표를 다시 돌려 이벤트 JSON에 적는다.
6. 종료 코드는 데모를 따른다: 0 성공(세 파일을 썼다), 2 실행할 수 없음(설정·모델·MediaPipe 모델 파일의 오류, 쓸 수 없는
   출력 폴더, 포트 사용 중, 열 수 없는 소스, 접속 시간 초과, 프레임을 하나도 받지 못함, 세션 도중의 처리 오류, 끝의
   그림·JSON 쓰기 실패).
7. 세션이 도는 동안에는 시작할 때 만든 객체(라이브러리, 모델)를 가비지 수거에서 빼 둔다(`gc.freeze()`. 끝나면 푼다).
   빼 두지 않으면 세션 도중의 전체 수거가 그 객체들(이 PC에서 308,341개)을 훑느라 프레임 하나의 처리가 그만큼 늦는다
   (아래 전제와 한계의 잰 값).

터미널 출력 (영어. 모두 표준 출력, 오류와 모델 대체 알림만 표준 오류)
    처음       무엇을 돌리는지(`startup_lines`): 도구 설명, 모델 파일(무작위 가중치면 그 사실), 졸음 판정원
               (`drowsiness.source`. 모델을 쓰는 설정인데 무작위 가중치면 졸음 이벤트에 뜻이 없다는 경고), 소스, 듣는
               소스면 보내는 쪽의 명령 예와 카메라 이름을 보는 명령, 회전(`--rotation`을 줬는지), 속도, 페이지 주소,
               끝내는 방법
    기다림     `WAIT_NOTE_INTERVAL_S`마다 `still waiting for the sender (90 s left)`(남은 시간이 0초로 적힐 때는 내지
               않는다)
    소스 열림  `source opened: the sender is connected and sends video; waiting for the first key frame`.
               `FIRST_FRAME_HINT_S`(3초)가 지나도 프레임이 없으면 키프레임 간격을 보라는 한 줄
    첫 프레임  `first frame: 640x480; the session clock starts here`, 이어서 기준선 안내 한 줄(`baseline_note`: 처음
               60초가 기준이고, 그동안 감김 이벤트는 임시이며 시선 이벤트는 나오지 않는다)
    1초마다    `format_status_line`: 세션 시간, 얼굴, closure, 전방 주시, 기준선 진행 또는 완성, 초당 받은·처리한
               프레임 수, 버린 수(`dropped`: 처리가 밀려서, `skipped`: 타임스탬프 때문에)
    이벤트     `event 21.5 s: microsleep onset (...)` 한 줄, 경고는 `alert 21.5 s: microsleep (immediate, ...)` 한 줄
    끝         멈추라는 요청이면 `stop requested: ending the session`, 그리고 `session ended: <이유>`, 프레임 계수 한 줄
               (`decoded = skipped + dropped + left + failed + processed`), `wrote <표>`, 이어지는 데모 출력의 회전 줄이 이
               세션에서 무엇을 뜻하는지 알리는 `replay: ...` 한 줄, 이어서 `python -m drowsy.demo --raw`의 출력 그대로

웹 페이지 (`StatusServer`. 표준 라이브러리 HTTP 서버, 127.0.0.1에만 묶는다)
    `/`              HTML 한 장(외부 자원 없음). `PAGE_POLL_MS`(500 ms)마다 `/status.json`을 읽어 그린다.
    `/status.json`   `phase`(`waiting`, `running`, `ended`), `source_open`(소스가 열렸는지), `run`(프로그램의 실행마다
                     다른 값), `hint`(기준선이 완성될 때까지의 기준선 안내), 상태 줄과 같은 값들, 최근 이벤트와 경고,
                     무엇을 돌리는지의 안내 줄(`run_lines`)
    `/preview.mjpg`  multipart JPEG 스트림. `--no-preview`면 404다.
- Host 머리말이 `localhost`, `127.0.0.1`, `[::1]`에 서버의 포트가 붙은 것이 아닌 요청은 403이다. 브라우저에 열린 다른
  웹 페이지가 자기 이름을 127.0.0.1로 돌려(DNS 재지정) 미리보기를 읽어 가는 길을 막는다. 머리말이 없는 요청은 받는다.
- 페이지는 프로그램이 답하지 않으면 미리보기 그림을 치우고, `run`이 바뀌면(프로그램을 다시 시작했다) 값과 이벤트를
  지우고 그림을 새로 만든다. 탭을 열어 둔 채 프로그램을 다시 시작해도 된다.
- 응답 도중에 끊긴 연결(새로 고침, 탭 닫기)은 터미널에 아무것도 남기지 않는다. 읽지 않는 연결은 `HTTP_TIMEOUT_S`(10초)
  뒤에 끊는다.

세션이 끝나면 쓰는 파일 (`--out` 아래. 줄기는 `live_<YYYYMMDD_HHMMSS>`, 세션을 시작한 지역 시각)
    `<줄기>.parquet`        세션의 프레임 표. 추출 결과와 같은 형식이고(`extract.write_raw_frames`) 랜드마크 값만 담는다.
                            메타데이터: `source`, `live: true`, `model_sha256`, `mediapipe_version`, `drop_sim: null`,
                            `rotation`과 `rotation_override`(둘 다 세션에 쓴 각도), `rotation_given`(`--rotation`을 줬는지),
                            `n_decoded`, `n_skipped`, `n_dropped`, `n_left`, `n_failed`, `n_decode_errors`,
                            `n_processed`, `n_face`, `width`, `height`, `arrival_clock`, `end_reason`.
                            `n_decoded = n_skipped + n_dropped + n_left + n_failed + n_processed`다. `n_dropped`는 처리가
                            밀려 버린 수, `n_left`는 세션이 끝날 때 처리하지 못하고 남은 수, `n_failed`는 처리하다 오류가
                            나 표에 넣지 못한 수다(처리 오류는 세션을 끝내므로 0이나 1). 추출 결과의 `n_dropped`는 프레임
                            누락 모사가 버린 수여서 뜻이 다르다.
    `<줄기>_timeline.png`   그 표에 `python -m drowsy.demo --raw`를 돌린 타임라인
    `<줄기>_events.json`    같은 실행의 이벤트 JSON. 터미널에 나온 이벤트와 경고가 그대로 들어 있다.

명령 (저장소 루트에서 실행)
    python -m drowsy.live [--source tcp://127.0.0.1:5599?listen=1] [--rotation 0] [--port 8765] [--no-preview]
                          [--model export/drowsiness_v0.onnx] [--speed-kmh 60] [--out out/live] [--config ...]

전제와 한계
- 보내는 쪽의 타임스탬프가 실제 촬영 간격을 반영해야 한다. 파이프라인의 시간은 모두 이 값에서 나온다.
- 영상은 H.264로 압축된 것이다(영상 파일에서 추출할 때와 같다). 압축하지 않은 카메라 프레임을 받는 앱과는 화소가 다르다.
- 모델 `drowsiness_v0.onnx`는 무작위 가중치다. 그 출력은 판정이 아니고, 뜻이 있는 것은 규칙 이벤트뿐이다(기본 규칙
  설정의 졸음 판정원은 PERCLOS다. 판정원을 모델로 바꾼 설정에서는 졸음 이벤트에도 뜻이 없고, 시작 줄이 그렇게 알린다).
  그래서 터미널과 웹 페이지에는 모델 출력을 보여 주지 않는다(이벤트 JSON과 그림에는 데모와 같이 들어간다).
- 데스크톱 시험 도구다. Android에 관해서는 아무것도 검증하지 않는다.
- 실제 카메라, Windows의 ffmpeg, 브라우저에 그려진 페이지, 얼굴이 있는 영상으로는 돌려 보지 못했다. 아래의 값은 모두
  이 PC 안에서 PyAV로 만든 보내는 쪽과 얼굴 없는 합성 그림으로 잰 것이다.
- 받는 쪽은 버퍼링을 끄고 읽으므로(`fflags=nobuffer`) 스트림 분석이 끝난 뒤의 첫 키프레임부터 본다. 키프레임 간격이
  30인 30 fps 스트림에서 처음 30프레임을 보지 못했다(2026-11-14, 이 PC의 localhost, PyAV 보내는 쪽).
- 보여 주는 상태는 실제보다 늦다. 보내는 쪽이 프레임을 인코더에 넣은 때부터 그 프레임의 처리가 끝날 때까지 30 fps
  스트림에서 중앙값 71.8 ms, 최대 87.6 ms였다(중앙값은 프레임 간격 둘에 조금 더한 값. 같은 날, 같은 조건, 얼굴 없는
  640x480 그림 6,000프레임(200초), 버린 프레임 0). 파이프라인의 시간은 스트림의 타임스탬프에서 나오므로 이 지연은
  판정에 들어가지 않는다.
- 얼굴이 있는 영상에서의 처리 속도는 재지 못했다(시험 그림에는 얼굴이 없다). 얼굴 없는 그림에서는 프레임 하나의
  처리(MediaPipe와 파이프라인)가 중앙값 1.33 ms, 최대 6.51 ms였다(같은 측정). 그 세션의 상주 메모리는 시작 5초 뒤
  793 MB, 끝(200초)에 805 MB였다.
- 세션 도중의 전체 수거(설계 결정 7): `gc.freeze()`를 넣기 전에는 프레임 하나가 60 ms 넘게 걸리는 일이 있었다
  (1,800프레임 가운데 하나가 68.4 ms). 그 프레임과 같은 순간에 시작한 2세대 전체 수거가 원인이었다(검토 때 75.0 ms짜리
  수거와 76.6 ms짜리 프레임, 넣기 직전의 실행에서 65.1 ms와 66.7 ms). 넣은 뒤에는 같은 실행 1,470프레임에서 20 ms를
  넘은 프레임이 없었고, 위의 6,000프레임에서도 가장 오래 걸린 프레임이 6.51 ms였다.

가정: 소스의 형식은 소스 문자열로 정한다(`plan_source`). `tcp://`와 `udp://`는 MPEG-TS와 지연을 줄이는 옵션
  (`STREAM_OPTIONS`)으로, `/dev/video`로 시작하면 v4l2로 열고, 그 밖은 PyAV가 내용을 보고 정하게 둔다. 형식을 고르는
  옵션은 두지 않았다(승인된 옵션 목록에 없다).
가정: 영상 파일 소스는 타임스탬프대로 실제 속도로 재생한다(첫 프레임을 낸 때부터 `t_ms`만큼 지난 때에 그 프레임을
  낸다). 파일은 디코딩이 실제 시간보다 훨씬 빨라서, 그대로 두면 가장 새 프레임만 넘기는 규칙이 대부분의 프레임을
  버린다. 파일 전체를 빠짐없이 돌리려면 `python -m drowsy.demo --video`를 쓴다.
가정: 소스의 회전 메타데이터와 표시 행렬은 보지 않는다. `--rotation`으로 준 각도(기본 0)만 쓴다.
가정: 첫 프레임에 pts가 없거나 스트림에 time_base가 없으면 스트림 전체를 도착 시각 시계로 읽는다. 첫 프레임에는
  pts가 있는데 뒤 프레임에 없으면 그 프레임은 버리고 센다(두 시계를 섞지 않는다. `video_io`와 같다).
가정: 풀 수 없는 패킷(`InvalidDataError`)은 건너뛰고 센다(`StreamCounts.decode_errors`). 스트림 하나의 깨진 패킷이
  세션을 끝내지 않게 하려는 것이다.
가정: 보내는 쪽의 ffmpeg 명령 예 가운데 카메라 입력 부분(`-f dshow -video_size ... -framerate ... -i video=...`과 목록을
  보는 두 명령)은 실행해 보지 못했다(Windows에 ffmpeg가 없고 Windows 프로그램을 실행하지 않았다). 옵션의 이름과 자리는
  검토 때 ffmpeg 공식 문서와 대조했다. 그 뒤의 인코더와 출력 부분(libx264, ultrafast, zerolatency, `-g 30`, yuv420p,
  MPEG-TS, TCP)은 PyAV로 만든 보내는 쪽과, 검토 때 이 PC의 Linux ffmpeg 7.0.2 명령으로 확인했다(합성 그림). 그 Linux
  ffmpeg 명령은 주소를 `tcp://127.0.0.1:<포트>`로 적었다: 그 빌드는 `localhost`라는 이름으로는 접속하지 못했다
  (2026-11-14의 점검에서 `tcp://localhost:<포트>`로는 12초 동안 접속하지 못하고 메시지 없이 끝났다).
가정: 보내는 쪽의 명령 예는 주소를 `localhost`로 적는다(`sender_target`). 2026-11-14에 Windows PowerShell의 TCP 접속이
  WSL 안의 듣는 쪽에 `localhost`로도 `127.0.0.1`로도 닿았다(웹 페이지 요청은 `localhost`로 닿았다). 영상은 Windows에서
  보내 본 적이 없으므로 Windows의 ffmpeg가 `localhost`로 접속하는지는 확인하지 못했다. 그래서 접속이 거부되거나 보내는
  쪽을 시작했는데도 `still waiting` 줄만 나오면 `127.0.0.1`로 바꿔 보라고 안내한다(`--help`).
가정: 듣는 소스에 접속한 쪽이 영상 없이 끊으면 오류로 끝내지 않고 남은 시간 동안 다시 듣는다(`open_source`). PyAV는
  접속을 받은 때를 알려 주지 않으므로 "source opened"는 스트림 분석까지 끝난 뒤에 나온다. 그래서 접속한 채 MPEG-TS가
  아닌 것을 계속 보내거나 아무것도 보내지 않는 쪽이 있어도 접속 시간 초과까지 `still waiting` 줄만 나온다. 시간 초과
  뒤의 오류 문장은 아무도 접속하지 않은 경우와 접속한 쪽이 MPEG-TS 영상을 보내지 않은 경우를 함께 적고
  `-f mpegts`를 알려 준다.
가정: 기다리는 시간들(`CONNECT_TIMEOUT_S`, `STALL_TIMEOUT_S`), 미리보기와 상태 줄의 주기, 프레임 상한은 옵션이 아니라
  이 모듈의 상수다(설정 파일과 승인된 옵션 목록을 바꾸지 않는다).
가정: `--model`을 주지 않았을 때의 모델 선택(설정의 `demo.model_path`, 그 파일이 없으면 `paths.export_dir`의
  `drowsiness_v0.onnx`와 표준 오류의 알림)과 `--speed-kmh`의 범위 검사는 데모와 같다. 데모에서는 이 둘이 `main`과
  `_demo` 안에 있어 부를 수 없으므로 같은 내용을 여기에 다시 적었다.
가정: `LiveSession.process_raw`는 `demo.run_pipeline`의 프레임 하나 분량과 같은 일을 한다(`DriverPipeline.update`,
  `AlertKinds`, `EventRecord`, `AlertRecord`, `ProbPoint`). `run_pipeline`은 입력이 끝나야 결과를 돌려주므로 프레임마다
  부를 수 없다. 두 길이 같은 결과를 내는지는 테스트가 확인한다.
가정: 기준선 진행(`baseline 12/60 s`)의 분모는 지금 마감이다. 파이프라인이 마감을 내주지 않으므로 spec의
  `baseline.duration_ms`, `extend_step_ms`, `max_duration_ms`로 다시 구한다(60초, 미달이면 30초씩 180초까지).
가정: 프레임 표의 메타데이터에서 끝나야 아는 값(프레임 계수, 끝난 이유)은 처리 루프가 끝날 때 채운다.
  `extract.write_raw_frames`가 메타데이터를 프레임을 다 받은 뒤에 읽는 것에 기댄다(테스트가 확인한다).
가정: 프레임 표에는 세션에 쓴 각도를 `rotation_override`에도 적는다(그 키의 뜻이 "소스의 회전 메타데이터 대신 쓴
  각도"이고, 실시간 경로는 메타데이터를 보지 않는다). `--rotation`을 주지 않은 세션도 0을 그렇게 적는다. 그래서 끝의 데모
  출력의 회전 줄은 "manual value recorded by the extraction"이고, `python -m drowsy.dataset`의 회전 요약도 실시간 표를
  수동 값으로 센다. 실시간 표에서 그 말은 "`--rotation`으로 준 값이거나 기본값 0이고, 아무것도 점검하지 않았다"는
  뜻이다. 줬는지는 `rotation_given`에 적고, 데모 출력 앞의 `replay: ...` 줄이 그 뜻을 알린다. 데모의 문장은
  `drowsy.demo`에 있어 여기서 바꾸지 않았다. 녹화 요약(`extract_video`가 적는 밝기, 프레임 간격 등)과
  `extract_version`은 적지 않는다.
가정: 프레임 하나를 처리하다 난 오류는 `Exception`이면 모두 "처리 오류"로 다룬다(그때까지의 프레임을 쓰고 2로 끝난다).
  어떤 종류의 오류든 긴 세션을 표 없이 끝내는 것보다 낫다고 봤다. 오류가 난 그 프레임은 표에 들어가지 않고 끝의
  프레임 계수에서 `failed`(메타데이터 `n_failed`)로 따로 센다. 파이프라인이 받아들인 뒤 줄을 내다 오류가 난 프레임은
  표에 들어가므로 `processed`로 센다.
가정: 상태 줄은 실제 시간으로 `STATUS_INTERVAL_S`(1초)마다 한 번 출력하고, 초당 프레임 수는 그 사이에 받은·처리한
  프레임 수를 지난 시간으로 나눈 값이다. 프레임이 오지 않는 동안에도 마지막 상태로 줄을 낸다(0.0 fps로 보인다).
가정: 출력 파일의 이름은 세션을 시작한 지역 시각으로 짓는다. 같은 초에 시작한 두 세션은 같은 이름을 쓴다(덮어쓴다).
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import gc
import json
import math
import os
import re
import signal
import sys
import textwrap
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import av
import cv2
import numpy as np

from drowsy import demo, export_onnx, extract
from drowsy.config import load_config
from drowsy.pipeline import DriverPipeline, ModelFn
from drowsy.rules import ZONE_IN, ZONE_OUT, ZONE_UNKNOWN
from drowsy.spec import PipelineSpec, RulesConfig, load_feature_spec, load_rules
from drowsy.types import DROWSY_SOURCE_PERCLOS, RawFrame
from drowsy.video_io import VideoError, normalize_rotation, pts_to_ms, rotate_upright

DEFAULT_SOURCE = "tcp://127.0.0.1:5599?listen=1"  # Windows의 보내는 쪽이 접속하는 주소(이 프로그램이 듣는다)
DEFAULT_PORT = 8765  # 웹 페이지의 포트
DEFAULT_OUT_SUBDIR = "live"  # `--out`을 주지 않으면 <demo.out_dir>/live
BIND_HOST = "127.0.0.1"  # 웹 페이지는 이 주소에만 묶는다. 바꾸는 옵션은 없다(미리보기가 이 PC 밖으로 나가지 않게)

STREAM_FORMAT = "mpegts"  # tcp://, udp:// 소스를 읽는 형식
# 지연을 줄이는 읽기 옵션(2026-11-14에 이 PC에서 PyAV 18.1.0으로 확인한 조합).
STREAM_OPTIONS: dict[str, str] = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "probesize": "32768",
    "analyzeduration": "0",
}
DEVICE_PREFIX = "/dev/video"
DEVICE_FORMAT = "v4l2"

CONNECT_TIMEOUT_S = 120.0  # 소스가 열릴 때까지(듣는 소스면 보내는 쪽이 접속할 때까지) 기다리는 시간
OPEN_GRACE_S = 0.5  # 그 시간이 지난 뒤 PyAV가 스스로 시간 초과를 알리기를 더 기다리는 시간. 지나면 기다리기를 그만둔다
WAIT_NOTE_INTERVAL_S = 30.0  # 소스가 열리기를 기다리는 동안 남은 시간을 알리는 주기
STALL_TIMEOUT_S = 10.0  # 읽기가 이만큼 막히면 세션을 끝낸다
FIRST_FRAME_HINT_S = 3.0  # 소스가 열린 뒤 이만큼 프레임이 없으면 키프레임 간격을 알려 준다
POLL_S = 0.25  # 처리 루프가 새 프레임을 기다리는 한 번의 길이(멈추라는 요청과 스트림의 끝을 이 간격으로 본다)
STOP_JOIN_S = 2.0  # 끝낼 때 읽기 스레드를 기다리는 시간(한 번만 기다린다)
STATUS_INTERVAL_S = 1.0  # 상태 줄을 출력하는 주기(실제 시간)
PREVIEW_INTERVAL_S = 0.1  # 미리보기 JPEG를 만드는 최소 간격
PREVIEW_MAX_SIDE_PX = 480  # 미리보기의 긴 변(픽셀). 넘으면 줄인다
PREVIEW_JPEG_QUALITY = 70
PAGE_POLL_MS = 500  # 웹 페이지가 /status.json을 읽는 주기
HTTP_TIMEOUT_S = 10.0  # 웹 페이지 연결 하나의 소켓 시간 제한(읽지 않는 클라이언트의 연결을 끊는다)
# 웹 페이지가 받아들이는 Host 머리말의 이름(포트는 서버의 포트여야 한다). 다른 이름으로 온 요청은 403이다
ALLOWED_HOST_NAMES: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
RECENT_ITEMS = 20  # 웹 페이지에 남기는 최근 이벤트와 경고의 수(각각)
MAX_SESSION_FRAMES = 216_000  # 한 세션이 처리하는 프레임 수의 상한(30 fps로 2시간). 이르면 세션을 끝내고 파일을 쓴다

PHASE_WAITING = "waiting"  # 소스가 열리기를, 또는 첫 프레임을 기다린다
PHASE_RUNNING = "running"
PHASE_ENDED = "ended"  # 세션이 끝났고 파일을 쓰는 중이다

END_INTERRUPTED = "interrupted (Ctrl+C)"
END_SIGNAL = "stopped by a signal ({name})"  # SIGTERM(kill), SIGHUP(터미널을 닫음)
END_OF_STREAM = "the source reached its end (the sender stopped)"
END_STALLED = f"no data from the source for {STALL_TIMEOUT_S:g} s"
END_FRAME_LIMIT = f"the session reached the frame limit ({MAX_SESSION_FRAMES} frames)"

NOTE_TOOL = (
    "drowsy.live: desktop test of the streaming pipeline on a camera stream. It verifies nothing about Android. "
    "Assistive guidance only, not a safety system."
)
NOTE_RANDOM_MODEL = "the model output carries no meaning, only the rule events do"
# 무작위 가중치 모델인데 규칙 설정의 졸음 판정원이 모델을 쓸 때(`drowsiness.source`가 model이나 either)
NOTE_RANDOM_MODEL_DROWSY = (
    "the model output carries no meaning, and neither do drowsy events (drowsiness.source {source} takes them from "
    "the model); only the other rule events do"
)

# 보내는 쪽의 예(Windows PowerShell의 ffmpeg). {target}은 이 프로그램이 듣는 주소다. 모듈 설명의 가정 참조
SENDER_COMMAND = (
    'ffmpeg -hide_banner -f dshow -video_size 640x480 -framerate 30 -i video="<camera name>" -c:v libx264 '
    "-preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p -f mpegts {target}"
)
SENDER_LIST_DEVICES = "ffmpeg -hide_banner -list_devices true -f dshow -i dummy"
SENDER_LIST_OPTIONS = 'ffmpeg -hide_banner -list_options true -f dshow -i video="<camera name>"'
# 끝내라는 뜻으로 받는 신호. 처리기는 표시만 세운다(SIGHUP은 터미널 창을 닫을 때 온다. Windows에는 없다)
_STOP_SIGNALS: tuple[int, ...] = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)
)

_ZONE_WORDS: dict[str, str] = {ZONE_IN: "forward", ZONE_OUT: "away", ZONE_UNKNOWN: "unknown"}
_MJPEG_BOUNDARY = b"drowsyliveframe"


class LiveError(RuntimeError):
    """실시간 세션을 진행할 수 없을 때(열리지 않는 소스, 접속 시간 초과, 사용 중인 포트, 프레임 없음 등)."""


# ---------------------------------------------------------------------------
# 소스: 열기, 프레임과 시각
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """소스 문자열을 여는 방법(`plan_source`의 결과)."""

    format: str | None  # PyAV에 넘기는 형식 이름. None이면 PyAV가 정한다
    options: dict[str, str]  # PyAV에 넘기는 옵션
    listening: bool  # 이 프로그램이 듣고 보내는 쪽이 접속하는 소스인지(`listen=1`)
    realtime: bool  # 타임스탬프대로 실제 속도로 재생할 소스인지(영상 파일)


def plan_source(source: str) -> SourcePlan:
    """소스 문자열로 여는 방법을 정한다(모듈 설명의 가정 참조)."""
    if source.startswith(("tcp://", "udp://")):
        listening = re.search(r"[?&]listen=1(&|$)", source) is not None
        return SourcePlan(STREAM_FORMAT, dict(STREAM_OPTIONS), listening, False)
    if source.startswith(DEVICE_PREFIX):
        return SourcePlan(DEVICE_FORMAT, {}, False, False)
    return SourcePlan(None, {}, False, Path(source).is_file())


def sender_target(source: str) -> str:
    """듣는 소스(`tcp://127.0.0.1:5599?listen=1`)에 보내는 쪽이 접속할 주소(`tcp://localhost:5599`).

    이 PC 자신을 가리키는 주소(127.0.0.1, 0.0.0.0, ::, ::1)는 `localhost`로 적는다(모듈 설명의 가정).
    """
    parts = urlsplit(source)
    host = parts.hostname or ""
    if host in ("", "127.0.0.1", "0.0.0.0", "::", "::1", "localhost"):
        host = "localhost"
    elif ":" in host:
        host = f"[{host}]"
    port = "" if parts.port is None else f":{parts.port}"
    return f"{parts.scheme}://{host}{port}"


_OPEN_TIMED_OUT = object()  # `_open_once`가 기다리기를 그만뒀다는 표시


def _open_once(
    source: str,
    plan: SourcePlan,
    timeout_s: float,
    say: Callable[[str], None] | None,
    note: Callable[[], str | None],
) -> Any:
    """PyAV로 소스를 한 번 연다. 컨테이너, PyAV가 낸 오류, 또는 `_OPEN_TIMED_OUT`을 돌려준다(오류를 내지 않는다).

    기다리는 동안 `WAIT_NOTE_INTERVAL_S`마다 `note()`가 돌려준 줄을 `say`로 낸다(None이면 그때는 내지 않는다).

    여는 일은 보조 스레드가 하고 호출한 스레드는 짧게 끊어 기다린다. PyAV의 열기는 C 안에서 막혀 있어서, 주 스레드가
    직접 부르면 Ctrl+C가 시간 초과까지 전달되지 않기 때문이다. `timeout_s + OPEN_GRACE_S`가 지나도록 PyAV가 돌아오지
    않으면 기다리기를 그만둔다(`udp://`는 PyAV의 시간 초과가 준 값의 두 배 뒤에 온다. 2026-11-14에 이 PC에서 5초를 주고
    10.11초). 그만두거나 중단되면(KeyboardInterrupt) 보조 스레드는 나중에 열린 컨테이너를 스스로 닫는다.
    """
    box: list[Any] = []
    lock = threading.Lock()
    abandoned = [False]

    def opener() -> None:
        try:
            container = av.open(
                source, format=plan.format, options=dict(plan.options), timeout=(timeout_s, STALL_TIMEOUT_S)
            )
        except BaseException as exc:  # 호출한 스레드가 받아서 가린다
            box.append(exc)
            return
        with lock:
            if abandoned[0]:  # 호출자는 이미 떠났다
                container.close()
            else:
                box.append(container)

    def abandon() -> bool:
        """기다리기를 그만둔다. 그 사이에 열기가 끝나 있었으면 False다(결과가 `box`에 있다)."""
        with lock:
            if box:
                return False
            abandoned[0] = True
            return True

    thread = threading.Thread(target=opener, name="drowsy-live-open", daemon=True)
    started_s = time.monotonic()
    next_note_s = started_s + WAIT_NOTE_INTERVAL_S
    thread.start()
    try:
        while thread.is_alive():
            thread.join(0.1)
            now_s = time.monotonic()
            if thread.is_alive() and now_s - started_s >= timeout_s + OPEN_GRACE_S and abandon():
                return _OPEN_TIMED_OUT
            if say is not None and thread.is_alive() and now_s >= next_note_s:
                next_note_s += WAIT_NOTE_INTERVAL_S
                text = note()
                if text is not None:
                    say(text)
    except BaseException:
        if not abandon() and not isinstance(box[0], BaseException):
            box[0].close()  # 중단되는 바로 그때 열렸다
        raise
    if not box:  # 보조 스레드가 `abandoned`를 보고 스스로 닫았다(여기에는 오지 않는다. 방어용)
        return _OPEN_TIMED_OUT
    return box[0]


def open_source(
    source: str, timeout_s: float = CONNECT_TIMEOUT_S, say: Callable[[str], None] | None = None
) -> Any:
    """소스를 열어 PyAV 입력 컨테이너를 돌려준다. 다 쓰면 호출자가 `close()`한다.

    - timeout_s: 열릴 때까지 기다리는 시간. 듣는 소스(`listen=1`)면 보내는 쪽이 접속해 영상을 보낼 때까지다.
      지나면 `LiveError`다(PyAV가 막혀 있어도 `OPEN_GRACE_S` 뒤에는 그만둔다). 연 뒤의 읽기는 `STALL_TIMEOUT_S` 동안
      데이터가 없으면 `av.error.ExitError`를 낸다.
    - 듣는 소스에 접속한 쪽이 영상 없이 끊거나(포트만 확인해 본 접속) MPEG-TS가 아닌 것을 보내고 끊으면, 남은 시간
      동안 다시 듣는다. 남은 시간이 없으면 할 일을 적은 `LiveError`다.
    - say: 한 줄을 출력하는 함수(없으면 아무것도 출력하지 않는다). 기다리는 동안 `WAIT_NOTE_INTERVAL_S`마다 남은
      시간을(0초로 적힐 알림은 내지 않는다), 다시 듣게 될 때 그 사실을 알린다.
    - 듣는 소스의 시간 초과 오류는 두 경우를 함께 적는다: 아무도 접속하지 않았거나, 접속한 쪽이 MPEG-TS 영상을 보내지
      않았다(둘을 가릴 수 없다. 모듈 설명의 가정).
    - 주소가 이미 쓰이고 있거나 비디오 스트림이 없으면 `LiveError`, 그 밖의 실패는 PyAV의 오류 그대로다.
    """
    plan = plan_source(source)
    deadline_s = time.monotonic() + timeout_s

    def left_s() -> float:
        return max(0.0, deadline_s - time.monotonic())

    def note() -> str | None:
        left = f"{left_s():.0f}"
        if left == "0":  # 시간이 다 됐다: "0 s left"라고 알리지 않는다(곧 시간 초과의 오류 문장이 나온다)
            return None
        if plan.listening:
            return f"still waiting for the sender ({left} s left)"
        return f"still opening {source} ({left} s left)"

    stream_fix = "Fix: the sender must send MPEG-TS (ffmpeg: -f mpegts) with H.264 video; see --help for the command."
    while True:
        result = _open_once(source, plan, max(left_s(), 0.001), say, note)
        if result is _OPEN_TIMED_OUT or isinstance(result, av.error.ExitError):
            cause = result if isinstance(result, BaseException) else None
            if plan.listening:
                # 접속한 채 MPEG-TS 영상을 보내지 않는 쪽도 여기로 온다(PyAV는 접속을 받은 때를 알려 주지 않는다)
                raise LiveError(
                    f"no video from a sender on {source} within {timeout_s:g} s: either no sender connected, or a "
                    "client stayed connected and sent no MPEG-TS video (ffmpeg: -f mpegts). Fix: start this program "
                    "first, then start the sender while it waits (ffmpeg on Windows sending MPEG-TS with H.264 to the "
                    "same port on localhost; see --help for the command)."
                ) from cause
            raise LiveError(f"{source} gave no video within {timeout_s:g} s") from cause
        if isinstance(result, OSError) and result.errno == errno.EADDRINUSE:
            raise LiveError(
                f"{source}: the address is already in use (another drowsy.live or another program listens there). "
                "Fix: stop that program, or use another port in --source and in the sender."
            ) from result
        if plan.listening and isinstance(result, (av.error.EOFError, av.error.InvalidDataError)):
            # 접속한 쪽이 영상 없이 끊었다(포트만 확인했거나, MPEG-TS가 아닌 것을 보냈다)
            if left_s() <= 0.0:
                raise LiveError(f"a client connected to {source} but sent no MPEG-TS video. {stream_fix}") from result
            if say is not None:
                say(
                    f"a client connected to {source} and closed without sending MPEG-TS video; listening again "
                    f"({left_s():.0f} s left). If that was the sender: it must send MPEG-TS (ffmpeg: -f mpegts) with "
                    "H.264 video."
                )
            continue
        if isinstance(result, BaseException):
            raise result
        if len(result.streams.video) == 0:
            result.close()
            if plan.format == STREAM_FORMAT:
                raise LiveError(f"no video stream in {source}. {stream_fix}")
            raise LiveError(f"no video stream in {source}")
        return result


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """소스에서 받은 프레임 하나: 세션 시각(양수 ms, 엄격히 증가)과 똑바로 세운 RGB uint8 [H, W, 3]."""

    t_ms: int
    rgb: np.ndarray


@dataclass(slots=True)
class StreamCounts:
    """`iter_live_frames`가 진행하면서 채우는 계수. 읽기 스레드가 쓰고 다른 스레드는 읽기만 한다."""

    decoded: int = 0  # 디코더가 내준 프레임 수
    skipped: int = 0  # 타임스탬프가 이전 값 이하이거나 pts가 없어서 버린 프레임 수
    decode_errors: int = 0  # 풀 수 없어 건너뛴 패킷 수
    arrival_clock: bool = False  # pts가 없어 도착 시각으로 시각을 만들었는지


def iter_live_frames(
    container: Any,
    rotation: int = 0,
    counts: StreamCounts | None = None,
    clock: Callable[[], float] = time.monotonic,
    stop: threading.Event | None = None,
) -> Iterator[LiveFrame]:
    """연 컨테이너의 프레임을 `LiveFrame`으로 하나씩 낸다. 스트림이 끝나면 끝난다.

    - `t_ms`는 스트림의 pts에서 `video_io.pts_to_ms`로 구한다: 첫 프레임이 1이고 엄격히 증가한다. 증가하지 않는 프레임과
      (pts가 있는 스트림에서) pts가 없는 프레임은 버리고 `counts.skipped`에 센다.
    - 첫 프레임에 pts가 없으면(또는 스트림에 time_base가 없으면) 도착 시각으로 만든다:
      `t_ms = 1 + floor((clock() − 첫 프레임의 clock())·1000 + 0.5)`. clock은 초를 돌려주는 단조 시계다.
    - rotation: 프레임을 반시계로 돌려 세우는 각도(90의 배수). 소스의 회전 메타데이터는 보지 않는다.
    - 풀 수 없는 패킷은 건너뛰고 `counts.decode_errors`에 센다. 읽기 오류(끊긴 연결, `STALL_TIMEOUT_S` 초과)는 PyAV의
      오류 그대로 난다.
    - stop: 세워지면 다음 패킷에서 끝난다(읽기 스레드를 멈출 때. 패킷은 오는데 프레임이 나오지 않는 동안, 곧 첫
      키프레임을 기다리는 동안에도 멈출 수 있게 패킷마다 본다. 데이터가 오지 않아 읽기에 막혀 있으면 보지 못한다).
    """
    if counts is None:
        counts = StreamCounts()
    turn = normalize_rotation(rotation)
    if len(container.streams.video) == 0:
        raise VideoError(f"no video stream in {container.name}")
    stream = container.streams.video[0]
    time_base = stream.time_base
    started = False
    first_pts = 0
    first_arrival = 0.0
    prev_ms = 0  # t_ms는 양수여야 하므로 0 이하인 값도 이 비교에서 걸러진다
    for packet in container.demux(stream):
        if stop is not None and stop.is_set():
            return
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            counts.decode_errors += 1
            continue
        for frame in frames:
            counts.decoded += 1
            first = not started
            if first:
                started = True
                counts.arrival_clock = frame.pts is None or time_base is None
                if not counts.arrival_clock:
                    first_pts = frame.pts
            if counts.arrival_clock:
                arrival = clock()
                if first:
                    first_arrival = arrival
                t_ms = 1 + int(math.floor((arrival - first_arrival) * 1000.0 + 0.5))
            elif frame.pts is None:
                counts.skipped += 1
                continue
            else:
                t_ms = pts_to_ms(frame.pts, first_pts, time_base)
            if t_ms <= prev_ms:
                counts.skipped += 1
                continue
            prev_ms = t_ms
            yield LiveFrame(t_ms, rotate_upright(frame.to_ndarray(format="rgb24"), turn))


class LatestFrameReader:
    """프레임을 읽는 스레드. 가장 새 프레임 하나만 들고 있다가 처리 루프에 넘긴다(모듈 설명의 설계 결정 1).

    - frames: `iter_live_frames`가 내는 것 같은 `LiveFrame`의 반복자. 디코딩은 이 스레드에서 일어난다.
    - realtime: 참이면 프레임을 타임스탬프대로 실제 속도로 낸다(영상 파일 소스). 첫 프레임을 낸 때부터
      `t_ms − 첫 t_ms`만큼 지난 때까지 기다린다.
    - `get(timeout_s)`: 가장 새 프레임을 가져간다. 그 시간 안에 프레임이 없으면 None이다.
    - `received`: 반복자에서 받은 프레임 수. `dropped`: 처리 루프가 가져가기 전에 더 새 프레임에 밀려 버린 프레임 수.
      처리 루프가 가져간 수는 `received − dropped`에서 아직 칸에 있는 프레임(0 또는 1)을 뺀 값이다.
    - `finished`: 반복자가 끝났고(또는 오류가 났고) 넘길 프레임도 남지 않았는지. `error`는 반복자가 낸 오류다(없으면 None).
    - stop: `stop()`이 세우는 표시. 같은 것을 `iter_live_frames`에도 주면 프레임이 나오지 않는 동안에도 반복자가 멈춘다.
    """

    def __init__(
        self,
        frames: Iterable[LiveFrame],
        realtime: bool = False,
        clock: Callable[[], float] = time.monotonic,
        stop: threading.Event | None = None,
    ) -> None:
        self._frames = iter(frames)
        self._realtime = realtime
        self._clock = clock
        self._cond = threading.Condition()
        self._slot: LiveFrame | None = None
        self._done = False
        self._received = 0
        self._dropped = 0
        self._stop = threading.Event() if stop is None else stop
        self._thread = threading.Thread(target=self._run, name="drowsy-live-reader", daemon=True)
        self.error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        start_s = 0.0
        first_ms: int | None = None
        try:
            for frame in self._frames:
                if self._realtime:
                    if first_ms is None:
                        first_ms = frame.t_ms
                        start_s = self._clock()
                    wait_s = start_s + (frame.t_ms - first_ms) / 1000.0 - self._clock()
                    if wait_s > 0.0 and self._stop.wait(wait_s):
                        break
                if self._stop.is_set():
                    break
                with self._cond:
                    self._received += 1
                    if self._slot is not None:
                        self._dropped += 1
                    self._slot = frame
                    self._cond.notify_all()
        except BaseException as exc:  # 읽기 오류는 처리 루프가 `error`로 읽는다
            self.error = exc
        finally:
            close = getattr(self._frames, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # 이미 끝난 반복자를 닫다 난 오류는 세션에 영향이 없다
                    pass
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def get(self, timeout_s: float) -> LiveFrame | None:
        with self._cond:
            self._cond.wait_for(lambda: self._slot is not None or self._done, timeout_s)
            frame, self._slot = self._slot, None
            return frame

    @property
    def received(self) -> int:
        with self._cond:
            return self._received

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    @property
    def finished(self) -> bool:
        with self._cond:
            return self._done and self._slot is None

    def stop(self, timeout_s: float = STOP_JOIN_S) -> bool:
        """스레드에 끝내라고 알리고 timeout_s까지 기다린다. 스레드가 끝났으면 True다.

        기다리는 것은 처음 부를 때 한 번뿐이다. 다시 부르면 기다리지 않고 지금 끝났는지만 돌려준다(처리 루프와 `main`이
        한 번씩 부른다. 읽기에 막힌 스레드를 두 번 기다리지 않게).
        스레드가 읽기에 막혀 있으면 다음 패킷이 오거나 `STALL_TIMEOUT_S`가 지나야 끝난다. False이면 호출자는
        컨테이너를 닫지 않는다(읽는 중인 컨테이너를 다른 스레드에서 닫을 수 없다. 데몬 스레드라 프로세스와 함께 끝난다).
        """
        first = not self._stop.is_set()
        self._stop.set()
        if first and self._thread.is_alive():
            self._thread.join(timeout_s)
        return not self._thread.is_alive()


# ---------------------------------------------------------------------------
# 세션: 프레임 → RawFrame → DriverPipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiveStatus:
    """프레임 하나를 처리한 뒤의 상태. 불변이다.

    앞의 값들은 `LiveSession`이 채운다. 끝의 넷(`fps_received`부터)은 스트림 쪽 값이어서 처리 루프가 채운다
    (`LiveSession`이 돌려준 것에서는 0이다).
    """

    t_ms: int  # 마지막으로 받아들인 프레임의 t_ms(없으면 0)
    session_ms: int  # 세션 시간 = t_ms − 처음 받아들인 프레임의 t_ms
    frames: int  # 파이프라인이 받아들인 프레임 수
    face: bool
    closure: float  # closure_valid가 False이면 뜻이 없다
    closure_valid: bool
    provisional: bool  # 기준선 완성 전(전역 상수로 계산한 closure)인지
    gaze_zone: str  # `DriverPipeline.gaze_zone`(`ZONE_IN`, `ZONE_OUT`, `ZONE_UNKNOWN`)
    baseline_status: str | None  # 기준선의 상태. 완성 전이면 None
    baseline_ms: int  # 완성됐으면 완성된 세션 시간(T_b), 아니면 지금 마감(60000, 연장되면 90000, ...)
    events: int  # 지금까지의 이벤트 수
    alerts: int  # 지금까지의 경고 수
    model_outputs: int  # 지금까지의 모델 출력 수
    new_events: tuple[demo.EventRecord, ...] = ()  # 이 프레임이 낸 이벤트
    new_alerts: tuple[demo.AlertRecord, ...] = ()  # 이 프레임에서 낸 경고(종류가 붙어 있다)
    new_probs: tuple[demo.ProbPoint, ...] = ()  # 이 프레임이 발행시킨 스텝의 모델 출력
    fps_received: float = 0.0  # 초당 받은 프레임 수(읽기 스레드가 시각을 받아들인 프레임)
    fps_processed: float = 0.0  # 초당 처리한 프레임 수(MediaPipe와 파이프라인을 지난 프레임)
    dropped: int = 0  # 처리가 밀려 버린 프레임 수(누적)
    skipped: int = 0  # 타임스탬프가 증가하지 않아 버린 프레임 수(누적)


def _baseline_deadline_ms(session_ms: int, spec: PipelineSpec) -> int:
    """기준선이 아직 완성되지 않은 세션의 지금 마감(세션 시간 ms). `BaselineEstimator`의 마감 순서와 같다."""
    cfg = spec.baseline
    deadline_ms = cfg.duration_ms
    while session_ms >= deadline_ms and deadline_ms < cfg.max_duration_ms:
        deadline_ms = min(deadline_ms + cfg.extend_step_ms, cfg.max_duration_ms)
    return deadline_ms


class LiveSession:
    """실시간 세션 하나: 프레임을 RawFrame으로 바꿔 `DriverPipeline`에 넣고 그 뒤의 상태를 돌려준다.

    - spec, rules, model, speed_kmh: `demo.run_pipeline`에 주는 것과 같다. 파이프라인도 같은 인자로 만든다(기준선을 주지
      않는다). speed_kmh는 세션 내내 같은 차량 속도이고 None이면 경고를 속도로 억제하지 않는다.
    - landmarker: `extract.make_landmarker`가 만든 VIDEO 모드 FaceLandmarker. None이면 `process_raw`만 쓸 수 있다.
      `close()`가 닫는다.
    - `process(frame)`: `detect`와 `process_raw`를 차례로 부른다. 타임스탬프가 증가하지 않아 버린 프레임이면 상태는
      그대로이고 `new_events`, `new_alerts`, `new_probs`만 비어 있다.
    세는 값: `skipped`(타임스탬프 때문에 버린 프레임), `face_frames`(얼굴이 검출된 프레임).
    프레임마다 쌓아 두는 것이 없다(세션이 길어져도 이 객체의 크기는 늘지 않는다).
    """

    def __init__(
        self,
        spec: PipelineSpec,
        rules: RulesConfig,
        model: ModelFn | None,
        speed_kmh: float | None = None,
        landmarker: Any = None,
    ) -> None:
        self._spec = spec
        self._pipe = DriverPipeline(spec, rules, None, None, model)  # `demo.run_pipeline`과 같은 인자
        self._kinds = demo.AlertKinds()
        self._speed_kmh = speed_kmh
        self._landmarker = landmarker
        self._last_detect_ms = 0  # MediaPipe에 넣은 마지막 t_ms. t_ms는 양수여야 하므로 0에서 시작한다
        self._origin_ms = 0
        self._events = 0
        self._alerts = 0
        self._model_outputs = 0
        self.skipped = 0
        self.face_frames = 0
        self._status = LiveStatus(
            t_ms=0,
            session_ms=0,
            frames=0,
            face=False,
            closure=0.0,
            closure_valid=False,
            provisional=True,
            gaze_zone=self._pipe.gaze_zone,
            baseline_status=None,
            baseline_ms=spec.baseline.duration_ms,
            events=0,
            alerts=0,
            model_outputs=0,
        )

    @property
    def status(self) -> LiveStatus:
        """마지막으로 처리한 프레임 뒤의 상태. 프레임을 받기 전에는 `frames`가 0인 빈 상태다."""
        return self._status

    @property
    def spec(self) -> PipelineSpec:
        return self._spec

    def detect(self, frame: LiveFrame) -> RawFrame | None:
        """프레임 하나를 MediaPipe에 넣어 RawFrame으로 돌려준다.

        `t_ms`가 앞서 넣은 값보다 크지 않으면 MediaPipe에 넣지 않고 버리며(None) `skipped`에 센다. VIDEO 모드는
        엄격히 증가하는 양수 타임스탬프만 받는다.
        """
        if self._landmarker is None:
            raise LiveError("this LiveSession was made without a landmarker; only process_raw can be used")
        if frame.t_ms <= self._last_detect_ms:
            self.skipped += 1
            return None
        self._last_detect_ms = frame.t_ms
        return extract.detect_frame(self._landmarker, frame.t_ms, frame.rgb, self._spec)

    def process_raw(self, raw: RawFrame) -> LiveStatus:
        """RawFrame 하나를 파이프라인에 넣고 그 뒤의 상태를 돌려준다(`demo.run_pipeline`의 프레임 하나 분량).

        파이프라인이 버린 프레임(`t_ms`가 직전에 받아들인 값 이하)이면 `skipped`에 세고 상태는 그대로 둔다.
        """
        out = self._pipe.update(raw, self._speed_kmh)
        if out is None:
            self.skipped += 1
            self._status = dataclasses.replace(self._status, new_events=(), new_alerts=(), new_probs=())
            return self._status
        state = out.state
        if self._pipe.frames_accepted == 1:
            self._origin_ms = state.t_ms
        if state.face:
            self.face_frames += 1
        new_events = tuple(demo.EventRecord(state.t_ms, event, event in out.alerts) for event in out.events)
        new_alerts = tuple(
            demo.AlertRecord(state.t_ms, alert, kind)
            for alert, kind in zip(out.alerts, self._kinds.classify(out.events, out.alerts))
        )
        new_probs = tuple(
            demo.ProbPoint(state.t_ms, step.t_end_ms, step.k, probs)
            for step, probs in zip(out.steps, out.step_probs)
            if probs is not None
        )
        self._events += len(new_events)
        self._alerts += len(new_alerts)
        self._model_outputs += len(new_probs)
        session_ms = state.t_ms - self._origin_ms
        baseline = self._pipe.baseline
        self._status = LiveStatus(
            t_ms=state.t_ms,
            session_ms=session_ms,
            frames=self._pipe.frames_accepted,
            face=state.face,
            closure=state.closure,
            closure_valid=state.closure_valid,
            provisional=state.provisional,
            gaze_zone=self._pipe.gaze_zone,
            baseline_status=None if baseline is None else baseline.status,
            baseline_ms=(
                _baseline_deadline_ms(session_ms, self._spec) if baseline is None else baseline.t_complete_ms
            ),
            events=self._events,
            alerts=self._alerts,
            model_outputs=self._model_outputs,
            new_events=new_events,
            new_alerts=new_alerts,
            new_probs=new_probs,
        )
        return self._status

    def process(self, frame: LiveFrame) -> LiveStatus:
        """프레임 하나를 MediaPipe와 파이프라인에 차례로 넣고 그 뒤의 상태를 돌려준다."""
        raw = self.detect(frame)
        if raw is None:
            self._status = dataclasses.replace(self._status, new_events=(), new_alerts=(), new_probs=())
            return self._status
        return self.process_raw(raw)

    def close(self) -> None:
        """랜드마커를 닫는다."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ---------------------------------------------------------------------------
# 터미널 줄과 상태 JSON
# ---------------------------------------------------------------------------


def format_status_line(status: LiveStatus) -> str:
    """1초에 한 번 출력하는 상태 한 줄(영어).

    `session 12.3 s, face yes, closure 0.07 (provisional), gaze forward, baseline 12/60 s, received 30.0 fps,
    processed 29.9 fps, dropped 0, skipped 0`
    closure가 유효하지 않으면 `closure n/a`, 기준선이 완성됐으면 `baseline ok (60 s)`처럼 상태와 완성 시각을 적는다.
    """
    if not status.closure_valid:
        closure = "n/a"
    elif status.provisional:
        closure = f"{status.closure:.2f} (provisional)"
    else:
        closure = f"{status.closure:.2f}"
    if status.baseline_status is None:
        baseline = f"{status.session_ms // 1000}/{status.baseline_ms // 1000} s"
    else:
        baseline = f"{status.baseline_status} ({status.baseline_ms // 1000} s)"
    return (
        f"session {status.session_ms / 1000.0:.1f} s, face {'yes' if status.face else 'no'}, closure {closure}, "
        f"gaze {_ZONE_WORDS.get(status.gaze_zone, status.gaze_zone)}, baseline {baseline}, "
        f"received {status.fps_received:.1f} fps, processed {status.fps_processed:.1f} fps, "
        f"dropped {status.dropped}, skipped {status.skipped}"
    )


def format_event_line(record: demo.EventRecord, origin_ms: int) -> str:
    """이벤트 한 줄(영어). 시각은 세션 시간(초)이다: 앞의 값은 이벤트를 낸 프레임, `started`는 상태가 시작된 때다."""
    event = record.event
    provisional = ", provisional" if event.provisional else ""
    return (
        f"event {(record.t_ms - origin_ms) / 1000.0:.1f} s: {event.type} {event.phase} (severity {event.severity}, "
        f"started {(event.t_start_ms - origin_ms) / 1000.0:.1f} s, duration {event.duration_ms / 1000.0:.1f} s, "
        f"confidence {event.confidence:.2f}, source {event.source}{provisional})"
    )


def format_alert_line(record: demo.AlertRecord, origin_ms: int) -> str:
    """경고 한 줄(영어). 괄호 안의 첫 값은 경고의 종류(`demo.ALERT_KINDS`)다."""
    event = record.event
    return (
        f"alert {(record.t_ms - origin_ms) / 1000.0:.1f} s: {event.type} ({record.kind}, severity {event.severity}, "
        f"duration {event.duration_ms / 1000.0:.1f} s)"
    )


def status_fields(status: LiveStatus) -> dict[str, Any]:
    """상태 JSON에 넣는 값: 상태 줄 그대로(`line`)와 그 줄을 이루는 값들."""
    return {
        "line": format_status_line(status),
        "t_ms": status.t_ms,
        "session_s": status.session_ms / 1000.0,
        "frames": status.frames,
        "face": status.face,
        "closure": status.closure if status.closure_valid else None,
        "provisional": status.provisional,
        "gaze_zone": status.gaze_zone,
        "baseline": {
            "done": status.baseline_status is not None,
            "status": status.baseline_status,
            "elapsed_s": status.session_ms // 1000,
            "target_s": status.baseline_ms // 1000,
        },
        "fps_received": status.fps_received,
        "fps_processed": status.fps_processed,
        "dropped": status.dropped,
        "skipped": status.skipped,
    }


def event_fields(record: demo.EventRecord, origin_ms: int) -> dict[str, Any]:
    """상태 JSON의 이벤트 하나. 이벤트 JSON의 `events` 항목과 같은 값에 세션 시간(`session_s`)과 터미널 줄(`line`)을 더한다."""
    return {
        "t_ms": record.t_ms,
        **dataclasses.asdict(record.event),
        "alert": record.alert,
        "session_s": (record.t_ms - origin_ms) / 1000.0,
        "line": format_event_line(record, origin_ms),
    }


def alert_fields(record: demo.AlertRecord, origin_ms: int) -> dict[str, Any]:
    """상태 JSON의 경고 하나. 이벤트 JSON의 `alerts` 항목과 같은 값에 세션 시간과 터미널 줄을 더한다."""
    return {
        "t_ms": record.t_ms,
        **dataclasses.asdict(record.event),
        "kind": record.kind,
        "session_s": (record.t_ms - origin_ms) / 1000.0,
        "line": format_alert_line(record, origin_ms),
    }


def baseline_note(spec: PipelineSpec) -> str:
    """첫 프레임 뒤에 한 번 알리는 기준선 안내(영어). 웹 페이지에도 기준선이 완성될 때까지 보인다."""
    cfg = spec.baseline
    return (
        f"baseline: the first {cfg.duration_ms // 1000} s of the session are the reference for this driver (extended "
        f"up to {cfg.max_duration_ms // 1000} s while too few usable frames were seen). Look ahead normally. Until the "
        "baseline is complete, eye-closure events are provisional and looking away gives no event."
    )


def run_lines(
    model: demo.ModelInfo,
    source: str,
    plan: SourcePlan,
    rotation: int,
    speed_kmh: float | None,
    timeout_s: float = CONNECT_TIMEOUT_S,
    *,
    rotation_given: bool = True,
    drowsiness_source: str = DROWSY_SOURCE_PERCLOS,
    opened: bool = False,
) -> list[str]:
    """무엇을 돌리는지 적은 여섯 줄(영어): 도구 설명, 모델, 졸음 판정원, 소스, 회전, 속도.

    터미널의 시작 줄(`startup_lines`)과 웹 페이지의 "This run"에 쓴다. opened가 참이면 소스 줄이 "기다리는 중"이 아니라
    "열렸다"로 바뀐다(세션이 도는 동안 웹 페이지에 보이는 줄). rotation_given이 거짓이면 `--rotation`을 주지 않아
    기본값 0을 쓴다는 뜻이다.
    """
    random_model = model.metadata.get("trained_on") == export_onnx.TRAINED_ON_NONE
    model_line = (
        f"model: {model.path} (version {model.metadata['model_version']}, "
        f"trained_on: {model.metadata.get('trained_on', 'unknown')})"
    )
    if drowsiness_source == DROWSY_SOURCE_PERCLOS:
        if random_model:
            model_line += f": {NOTE_RANDOM_MODEL}"
        drowsy_line = "drowsiness.source: perclos (drowsy events come from PERCLOS, not from the model)"
    else:
        if random_model:
            model_line += f": {NOTE_RANDOM_MODEL_DROWSY.format(source=drowsiness_source)}"
        drowsy_line = f"drowsiness.source: {drowsiness_source} (drowsy events come from the model output"
        drowsy_line += ")" if drowsiness_source != "either" else " or from PERCLOS)"
        if random_model:
            drowsy_line += " WARNING: this model has random weights, so drowsy events carry no meaning"
    if plan.listening:
        how = "listening; the sender is connected" if opened else (
            f"listening; waiting up to {timeout_s:g} s for the sender to connect"
        )
    elif plan.realtime:
        how = "video file, played back at its own frame timing"
    else:
        how = "opened" if opened else f"opening, up to {timeout_s:g} s"
    turned = "from --rotation" if rotation_given else "the default; --rotation was not given"
    speed = "not given (alerts are not speed-gated)" if speed_kmh is None else f"{speed_kmh:g} km/h (constant)"
    return [
        NOTE_TOOL,
        model_line,
        drowsy_line,
        f"source: {source} ({how})",
        f"rotation: {rotation} ({turned}; a live stream has no automatic rotation check: look at the preview)",
        f"speed: {speed}",
    ]


def startup_lines(
    model: demo.ModelInfo,
    source: str,
    plan: SourcePlan,
    rotation: int,
    speed_kmh: float | None,
    page_url: str,
    preview: bool,
    timeout_s: float = CONNECT_TIMEOUT_S,
    *,
    rotation_given: bool = True,
    drowsiness_source: str = DROWSY_SOURCE_PERCLOS,
) -> list[str]:
    """시작할 때 출력하는 줄(영어): `run_lines`의 여섯 줄, 듣는 소스면 소스 줄 다음에 보내는 쪽의 명령 예 두 줄, 끝에
    페이지 주소와 끝내는 방법(Ctrl+C, 또는 듣는 소스면 보내는 쪽을 멈춤, 영상 파일이면 끝을 기다림, 그 밖의 소스면 소스를
    멈춤)."""
    lines = run_lines(
        model,
        source,
        plan,
        rotation,
        speed_kmh,
        timeout_s,
        rotation_given=rotation_given,
        drowsiness_source=drowsiness_source,
    )
    if plan.listening:
        lines[4:4] = [
            "sender example (Windows PowerShell; start it while this program waits): "
            + SENDER_COMMAND.format(target=sender_target(source)),
            f"camera names for the sender: {SENDER_LIST_DEVICES}",
        ]
    shown = (
        "status and camera preview; the preview is kept in memory only and served on 127.0.0.1 only"
        if preview
        else "status only; the camera preview is off (--no-preview)"
    )
    if plan.listening:
        ends = "stop the sender"
    elif plan.realtime:  # 영상 파일은 끝이 있다
        ends = "wait for the source to end"
    else:  # 카메라 장치, 접속하는 스트림 등: 끝을 기다릴 수 없다
        ends = "stop the source"
    lines += [
        f"page: {page_url} ({shown})",
        f"stop: press Ctrl+C in this terminal, or {ends}. The files of the session are written then; keep this "
        'window open until the "wrote" lines have appeared.',
    ]
    return lines


# ---------------------------------------------------------------------------
# 웹 페이지
# ---------------------------------------------------------------------------

PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>drowsy live</title>
<style>
:root { color-scheme: light dark; --ink: #0b0b0b; --soft: #52514e; --line: #d5d4cc; --bg: #fcfcfb; --card: #f0efec;
        --mark: #2a78d6; --warn: #b4460f; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #f0efec; --soft: #a9a8a1; --line: #3b3a37; --bg: #151514; --card: #222220; --mark: #86b6ef;
          --warn: #f0935f; }
}
body { margin: 0; padding: 16px 20px 28px; background: var(--bg); color: var(--ink);
       font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 20px; margin: 0 0 2px; }
h2 { font-size: 15px; margin: 22px 0 6px; }
p { margin: 4px 0; }
.soft { color: var(--soft); font-size: 13px; }
#phase { font-weight: 600; }
#phase.problem { color: var(--warn); }
.cols { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 12px; align-items: flex-start; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(150px, 1fr)); gap: 8px; flex: 1 1 320px; }
.cell { background: var(--card); border-radius: 8px; padding: 8px 12px; }
.cell .name { color: var(--soft); font-size: 12px; }
.cell .value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.cell .value.on { color: var(--mark); }
.cell .value.off { color: var(--warn); }
.bar { height: 6px; border-radius: 3px; background: var(--line); margin-top: 6px; overflow: hidden; }
.bar div { height: 100%; width: 0; background: var(--mark); }
#preview-box { flex: 1 1 320px; max-width: 480px; }
#preview-box img { display: block; width: 100%; border-radius: 8px; background: var(--card); min-height: 120px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 10px 4px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
th { color: var(--soft); font-weight: 500; }
pre { margin: 6px 0 0; white-space: pre-wrap; font: 12px/1.5 ui-monospace, Consolas, monospace; color: var(--soft); }
</style>
</head>
<body>
<h1>drowsy live</h1>
<p id="phase">connecting to the program ...</p>
<p class="soft" id="hint"></p>
<div class="cols">
  <div class="grid">
    <div class="cell"><div class="name">session time</div><div class="value" id="session">-</div></div>
    <div class="cell"><div class="name">face</div><div class="value" id="face">-</div></div>
    <div class="cell"><div class="name">eye closure (0 = open, 1 = closed)</div><div class="value" id="closure">-</div>
      <div class="bar"><div id="closure-bar"></div></div></div>
    <div class="cell"><div class="name">gaze zone</div><div class="value" id="gaze">-</div></div>
    <div class="cell"><div class="name">baseline</div><div class="value" id="baseline">-</div>
      <div class="bar"><div id="baseline-bar"></div></div></div>
    <div class="cell"><div class="name">frames per second (received / processed)</div>
      <div class="value" id="fps">-</div></div>
    <div class="cell"><div class="name">dropped (processing was behind)</div>
      <div class="value" id="dropped">-</div></div>
    <div class="cell"><div class="name">skipped (timestamp did not increase)</div>
      <div class="value" id="skipped">-</div></div>
  </div>
  <div id="preview-box">
    <p class="soft" id="preview-note">camera preview</p>
  </div>
</div>
<h2>Recent events</h2>
<table><thead><tr><th>at</th><th>event</th><th>phase</th><th>severity</th><th>duration</th><th>source</th>
<th>note</th></tr></thead><tbody id="events"></tbody></table>
<h2>Recent alerts</h2>
<table><thead><tr><th>at</th><th>alert</th><th>kind</th><th>severity</th><th>duration</th></tr></thead>
<tbody id="alerts"></tbody></table>
<h2>This run</h2>
<pre id="info"></pre>
<pre id="line"></pre>
<script>
const POLL_MS = __POLL_MS__;
const ZONES = {in: "forward", out: "away", unknown: "unknown"};
const VALUE_IDS = ["session", "face", "closure", "gaze", "baseline", "fps", "dropped", "skipped"];
function phaseText(s) {
  if (s.phase === "waiting") {
    return s.source_open ? "the source is open; waiting for the first picture" : "waiting for the camera stream";
  }
  if (s.phase === "running") { return "running"; }
  if (s.phase === "ended") {
    return "session ended" + (s.end_reason ? ": " + s.end_reason : "") +
      ". The files are being written (see the terminal).";
  }
  return String(s.phase);
}
const $ = (id) => document.getElementById(id);
let previewImg = null;  // 미리보기 스트림의 <img>. 프로그램의 실행마다 새로 만든다
let run = null;         // 페이지가 보여 주는 프로그램의 실행(`run` 값)

function setValue(id, text, cls) {
  const node = $(id);
  node.textContent = text;
  node.className = "value" + (cls ? " " + cls : "");
}

function seconds(value) { return value.toFixed(1) + " s"; }

function fillRows(id, items, cells) {
  const body = $(id);
  body.replaceChildren();
  for (const item of items.slice().reverse()) {
    const row = document.createElement("tr");
    for (const text of cells(item)) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
}

function dropPreview() {
  if (previewImg !== null) {
    previewImg.remove();
    previewImg = null;
  }
}

function clearValues() {
  for (const id of VALUE_IDS) { setValue(id, "-"); }
  $("closure-bar").style.width = "0%";
  $("baseline-bar").style.width = "0%";
  $("line").textContent = "";
  $("events").replaceChildren();
  $("alerts").replaceChildren();
}

function render(s) {
  const id = s.run === undefined ? "" : s.run;
  if (id !== run) {  // 첫 답이거나, 이 탭을 열어 둔 채 프로그램을 다시 시작했다
    dropPreview();
    clearValues();
    run = id;
  }
  const phase = $("phase");
  phase.className = "";
  phase.textContent = phaseText(s);
  $("hint").textContent = s.hint || "";
  $("info").textContent = (s.info || []).join("\\n");
  if (!s.preview) {
    dropPreview();
    $("preview-note").textContent = "the camera preview is off (--no-preview, or it failed: see the terminal)";
  } else if (previewImg === null) {
    const img = document.createElement("img");
    img.alt = "camera preview";
    img.onerror = () => { if (previewImg === img) { dropPreview(); } };  // 다음 상태가 새 그림을 만든다
    img.src = "/preview.mjpg?run=" + encodeURIComponent(String(id)) + "&t=" + Date.now();
    $("preview-box").prepend(img);
    previewImg = img;
    $("preview-note").textContent = "camera preview: the picture as the face landmarker gets it (not mirrored). " +
      "It is kept in memory only and never written to disk.";
  }
  if (s.session_s === undefined) {
    clearValues();
    return;
  }
  setValue("session", seconds(s.session_s));
  setValue("face", s.face ? "found" : "not found", s.face ? "on" : "off");
  setValue("closure", s.closure === null ? "n/a" : s.closure.toFixed(2) + (s.provisional ? " (provisional)" : ""));
  $("closure-bar").style.width = (s.closure === null ? 0 : Math.min(1, Math.max(0, s.closure)) * 100) + "%";
  setValue("gaze", ZONES[s.gaze_zone] || s.gaze_zone, s.gaze_zone === "out" ? "off" : "");
  const b = s.baseline;
  setValue("baseline", b.done ? b.status + " (" + b.target_s + " s)" : b.elapsed_s + " / " + b.target_s + " s");
  $("baseline-bar").style.width = (b.done ? 100 : Math.min(100, 100 * b.elapsed_s / b.target_s)) + "%";
  setValue("fps", s.fps_received.toFixed(1) + " / " + s.fps_processed.toFixed(1));
  setValue("dropped", String(s.dropped));
  setValue("skipped", String(s.skipped));
  $("line").textContent = s.line;
  fillRows("events", s.events, (e) => [seconds(e.session_s), e.type, e.phase, String(e.severity),
    seconds(e.duration_ms / 1000), e.source, (e.provisional ? "provisional " : "") + (e.alert ? "alert" : "")]);
  fillRows("alerts", s.alerts, (a) => [seconds(a.session_s), a.type, a.kind, String(a.severity),
    seconds(a.duration_ms / 1000)]);
}

async function poll() {
  try {
    const answer = await fetch("/status.json", {cache: "no-store"});
    render(await answer.json());
  } catch (error) {
    const phase = $("phase");
    phase.className = "problem";
    phase.textContent = "no answer from the program (it has ended or was stopped). " +
      "This page continues by itself when the program is started again on the same port.";
    $("hint").textContent = "";
    dropPreview();  // 프로그램이 사라진 뒤에는 그림을 남기지 않는다
    run = null;     // 다음 답은 새 실행이다: 그림과 값을 다시 만든다
  }
}

poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
""".replace("__POLL_MS__", str(PAGE_POLL_MS))


def encode_preview(rgb: np.ndarray) -> bytes:
    """세운 RGB 프레임 [H, W, 3]을 미리보기 JPEG 바이트로 만든다. 디스크에는 쓰지 않는다.

    긴 변이 `PREVIEW_MAX_SIDE_PX`를 넘으면 그 크기로 줄인다(가로세로 비율 유지). 품질은 `PREVIEW_JPEG_QUALITY`다.
    """
    height, width = rgb.shape[0], rgb.shape[1]
    scale = PREVIEW_MAX_SIDE_PX / max(height, width)
    if scale < 1.0:
        size = (max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    if not ok:
        raise LiveError("the preview picture could not be encoded as JPEG")
    return encoded.tobytes()


class _Handler(BaseHTTPRequestHandler):
    """`StatusServer`의 요청 처리기. GET만 받는다.

    - Host 머리말이 있으면 `localhost`, `127.0.0.1`, `[::1]` 가운데 하나에 서버의 포트가 붙은 것이어야 한다. 아니면
      403이다(다른 이름을 127.0.0.1로 돌려 놓고 브라우저에서 미리보기를 읽어 가는 것을 막는다). 머리말이 없는 요청은 받는다.
    - 응답을 쓰는 도중에 상대가 끊으면(새로 고침, 탭 닫기) 조용히 끝낸다(`_HTTPServer.handle_error`).
    """

    server: _HTTPServer
    timeout = HTTP_TIMEOUT_S  # 연결의 소켓 시간 제한: 요청을 보내지 않거나 응답을 읽지 않는 연결을 끊는다

    def do_GET(self) -> None:  # noqa: N802 (http.server가 정한 이름)
        board = self.server.board
        host = self.headers.get("Host")
        if host is not None and host.strip().lower() not in self.server.allowed_hosts:
            self._send(403, "text/plain; charset=utf-8", b"forbidden: this page answers only on localhost\n")
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE_HTML.encode("utf-8"))
        elif path == "/status.json":
            self._send(200, "application/json", board.status_bytes())
        elif path == "/preview.mjpg":
            if board.preview:
                self._stream_preview(board)
            else:
                self._send(404, "text/plain; charset=utf-8", b"the camera preview is off (--no-preview)\n")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        # 상대가 응답을 받기 전에 끊으면 OSError가 난다. 서버의 `handle_error`가 조용히 넘긴다.
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_preview(self, board: StatusServer) -> None:
        try:
            self.send_response(200)
            boundary = _MJPEG_BOUNDARY.decode("ascii")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            while True:
                sequence, jpeg = board.wait_preview(sequence)
                if jpeg is None:  # 서버가 멈췄다
                    return
                head = b"--%b\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (
                    _MJPEG_BOUNDARY,
                    len(jpeg),
                )
                self.wfile.write(head + jpeg + b"\r\n")
                self.wfile.flush()
        except OSError:  # 브라우저가 그림을 닫았거나, 읽지 않아서 시간 제한에 걸렸다
            self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (http.server가 정한 이름)
        """요청 기록을 터미널에 쓰지 않는다(상태 줄이 묻힌다)."""


class _HTTPServer(ThreadingHTTPServer):
    """요청마다 데몬 스레드를 쓰는 HTTP 서버. 처리기가 `board`로 `StatusServer`를 읽는다."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], board: StatusServer) -> None:
        super().__init__(address, _Handler)
        self.board = board
        port = int(self.server_address[1])
        self.allowed_hosts = frozenset(f"{name}:{port}" for name in ALLOWED_HOST_NAMES) | (
            frozenset(ALLOWED_HOST_NAMES) if port == 80 else frozenset()  # 포트 80이면 브라우저가 포트를 적지 않는다
        )

    def handle_error(self, request: Any, client_address: Any) -> None:
        """끊긴 연결(OSError)은 터미널에 traceback을 남기지 않고 넘긴다. 그 밖의 오류는 표준 동작대로 표준 오류에 적는다."""
        if isinstance(sys.exc_info()[1], OSError):
            return
        super().handle_error(request, client_address)


class StatusServer:
    """상태와 미리보기를 내주는 HTTP 서버. `BIND_HOST`(127.0.0.1)에만 묶는다.

    - 만들 때 포트에 묶는다. 포트가 쓰이고 있으면 OSError다. port가 0이면 운영체제가 빈 포트를 고른다(`port`로 읽는다).
    - preview가 False이면 `/preview.mjpg`는 404이고 `publish_preview`는 아무것도 하지 않는다.
    - `publish(payload)`: 상태 JSON을 바꾼다. `publish_preview(jpeg)`: 미리보기 그림을 바꾼다. 둘 다 처리 루프가 부르고,
      HTTP 스레드는 잠금 안에서 그 바이트의 참조만 읽는다(모듈 설명의 설계 결정 2).
    - 상태 JSON에는 언제나 `run`이 들어간다(`run_id`. 서버를 만들 때 정한 값). 웹 페이지는 이 값이 바뀌면 프로그램이
      다시 시작된 것으로 보고 미리보기 그림과 값을 새로 만든다.
    - `start()`로 서버 스레드를 시작하고 `stop()`으로 멈춘다(미리보기를 보내던 연결도 끝낸다).
    """

    def __init__(self, port: int = DEFAULT_PORT, preview: bool = True) -> None:
        self.preview = preview
        self.run_id = f"{os.getpid()}-{time.time_ns():x}"
        self._cond = threading.Condition()
        self._status = self._encode({"phase": PHASE_WAITING, "preview": preview})
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._closed = False
        self._httpd = _HTTPServer((BIND_HOST, port), self)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        """브라우저에 넣을 주소. Windows에서도 같은 주소로 열린다(WSL2가 localhost를 넘겨 준다)."""
        return f"http://localhost:{self.port}/"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.1}, name="drowsy-live-http", daemon=True
        )
        self._thread.start()

    def _encode(self, payload: Mapping[str, Any]) -> bytes:
        return json.dumps({**payload, "run": self.run_id}, allow_nan=False).encode("utf-8")

    def publish(self, payload: Mapping[str, Any]) -> None:
        encoded = self._encode(payload)
        with self._cond:
            self._status = encoded

    def status_bytes(self) -> bytes:
        with self._cond:
            return self._status

    def publish_preview(self, jpeg: bytes) -> None:
        if not self.preview:
            return
        with self._cond:
            self._jpeg = jpeg
            self._sequence += 1
            self._cond.notify_all()

    def wait_preview(self, sequence: int) -> tuple[int, bytes | None]:
        """sequence보다 새 그림이 올 때까지 기다려 (그 번호, 그림)을 돌려준다. 서버가 멈추면 (sequence, None)이다."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._sequence != sequence)
            if self._closed:
                return sequence, None
            return self._sequence, self._jpeg

    def stop(self) -> None:
        with self._cond:
            self._closed = True
            self._jpeg = None
            self._cond.notify_all()
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join()
            self._thread = None
        self._httpd.server_close()


# ---------------------------------------------------------------------------
# 처리 루프
# ---------------------------------------------------------------------------

# 시작할 때 이 오류들은 `error: ...` 한 줄과 종료 코드 2가 된다(데모와 같은 목록에 LiveError를 더했다.
# SpecError, ContractError, VideoError와 pyarrow의 형식 오류는 ValueError다)
_FATAL_ERRORS = (LiveError, demo.DemoError, extract.ExtractError, av.error.FFmpegError, ValueError, OSError)
# 이 오류들은 문장이 스스로 뜻을 전한다. 그 밖의 오류를 알릴 때는 종류 이름을 앞에 붙인다(`_error_text`)
_MESSAGE_ERRORS = (RuntimeError, ValueError, OSError, av.error.FFmpegError)


def _error_text(error: BaseException) -> str:
    """오류를 한 줄에 적을 문장. 문장이 없거나 종류를 알아야 읽히는 오류(KeyError, TypeError 등)는 종류 이름을 붙인다."""
    text = str(error)
    if text and isinstance(error, _MESSAGE_ERRORS):
        return text
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _say(text: str) -> None:
    """한 줄을 표준 출력에 바로 낸다. 터미널이 사라졌으면(창을 닫아 SIGHUP이 온 뒤, 닫힌 파이프) 그 줄은 버린다:
    출력할 곳이 없어도 세션의 파일은 써야 한다."""
    try:
        print(text, flush=True)
    except OSError:
        pass


def _complain(text: str) -> None:
    """오류 한 줄을 표준 오류에 낸다. 터미널이 사라졌으면 버린다."""
    try:
        print(f"error: {text}", file=sys.stderr, flush=True)
    except OSError:
        pass


class LiveLoop:
    """처리 루프: 읽기 스레드에서 프레임을 받아 세션에 넣고, 터미널 줄과 웹 페이지의 상태를 낸다.

    `frames()`는 처리한 프레임의 RawFrame을 하나씩 내는 생성기다. `extract.write_raw_frames`에 넘기면 그 함수가 프레임을
    하나 달라고 할 때마다 루프가 프레임 하나를 처리한다(모듈 설명의 설계 결정 5). 세션이 끝나면 생성기가 끝나고
    `metadata`에 끝나야 아는 값(프레임 계수, 끝난 이유)이 채워져 있다. 프레임을 하나도 처리하지 못했으면 `LiveError`다.

    - stop: 세워지면 처리 중인 프레임을 마치고 끝낸다(Ctrl+C 등). 세워진 뒤에는 새 프레임을 시작하지 않는다: 기다리던
      프레임이 그 뒤에 도착해도 처리하지 않는다. stop_reason은 그때 끝난 이유로 적을 문장을 돌려준다.
    - info: 웹 페이지에 보여 줄 안내 줄. say: 한 줄을 출력하는 함수.
    - 프레임 하나를 처리하다 난 오류는 종류를 가리지 않고(Exception) 세션을 끝내되, 그때까지의 프레임은 표에 남는다
      (`failure`). 미리보기를 만들다 난 오류는 미리보기만 끄고, 상태 JSON을 내다 난 오류는 한 번 알리고 세션을 계속한다.
    끝난 뒤에 읽는 값: `reason`(끝난 이유, 영어 문장), `failure`(세션을 끝낸 처리 오류. 없으면 None), `processed`,
    `failed`(처리하다 오류가 나 표에 넣지 못한 프레임 수: 0이나 1).
    끝의 프레임 계수는 `decoded = skipped + dropped + left + failed + processed`가 되게 센다(`left`는 나머지: 칸에 남은
    프레임, 멈추라는 요청 뒤에 받아 시작하지 않은 프레임, 읽기 스레드가 멈추기 직전에 디코딩한 프레임).
    """

    def __init__(
        self,
        reader: LatestFrameReader,
        session: LiveSession,
        server: StatusServer,
        counts: StreamCounts,
        stop: threading.Event,
        metadata: dict[str, Any],
        info: Sequence[str] = (),
        say: Callable[[str], None] = _say,
        clock: Callable[[], float] = time.monotonic,
        stop_reason: Callable[[], str] = lambda: END_INTERRUPTED,
    ) -> None:
        self._reader = reader
        self._session = session
        self._server = server
        self._counts = counts
        self._stop = stop
        self._stop_reason = stop_reason
        self._metadata = metadata
        self._info = list(info)
        self._say = say
        self._clock = clock
        self._events: deque[dict[str, Any]] = deque(maxlen=RECENT_ITEMS)
        self._alerts: deque[dict[str, Any]] = deque(maxlen=RECENT_ITEMS)
        self._fps_received = 0.0
        self._fps_processed = 0.0
        self._origin_ms = 0
        self._width = 0
        self._height = 0
        self._hint = ""  # 웹 페이지에 보일 안내(기준선이 완성될 때까지의 기준선 안내)
        self._preview_on = server.preview  # 미리보기를 만들다 오류가 나면 꺼진다
        self._last_preview_s = -math.inf
        self._page_failed = False
        self.processed = 0
        self.failed = 0
        self.reason: str | None = None
        self.failure: BaseException | None = None

    def status(self) -> LiveStatus:
        """세션의 상태에 스트림 쪽 값(초당 프레임 수, 버린 수)을 채운 것."""
        return dataclasses.replace(
            self._session.status,
            fps_received=self._fps_received,
            fps_processed=self._fps_processed,
            dropped=self._reader.dropped,
            skipped=self._counts.skipped + self._session.skipped,
        )

    def payload(self, phase: str) -> dict[str, Any]:
        """상태 JSON의 내용. `source_open`은 소스가 열렸다는 뜻이다(이 루프는 소스가 열린 뒤에 돈다)."""
        payload: dict[str, Any] = {
            "phase": phase,
            "preview": self._preview_on,
            "source_open": True,
            "info": self._info,
            "end_reason": self.reason,
            "hint": "",
        }
        if self.processed > 0:
            status = self.status()
            payload.update(status_fields(status))
            if status.baseline_status is None and phase == PHASE_RUNNING:
                payload["hint"] = self._hint
        payload["events"] = list(self._events)
        payload["alerts"] = list(self._alerts)
        return payload

    def _publish(self, phase: str) -> None:
        """상태 JSON을 낸다. 웹 페이지는 표시일 뿐이므로, 내지 못하면 한 번 알리고 세션은 계속한다."""
        try:
            self._server.publish(self.payload(phase))
        except Exception as exc:
            if not self._page_failed:
                self._page_failed = True
                self._say(f"page: the status could not be updated ({_error_text(exc)}); the session continues")

    def _fail(self, error: Exception) -> None:
        self.failure = error
        self.reason = f"processing failed ({_error_text(error)})"

    def _process(self, frame: LiveFrame) -> RawFrame | None:
        """프레임 하나를 세션에 넣고 그 프레임의 줄(첫 프레임, 이벤트, 경고)을 낸다. 표에 넣을 RawFrame을 돌려준다.

        타임스탬프 때문에 버린 프레임과 처리하다 오류가 난 프레임은 None이다(오류면 `failure`와 `reason`이 채워지고 그
        프레임을 `failed`에 센다). 파이프라인이 받아들인 프레임은 그 뒤에 줄을 내다 오류가 나도 돌려준다(표와 파이프라인이
        같은 프레임을 갖게. 그 프레임은 `processed`에 센다).
        """
        try:
            raw = self._session.detect(frame)
            if raw is None:
                return None
            status = self._session.process_raw(raw)
        except Exception as exc:  # MediaPipe나 파이프라인이 낸 어떤 오류든: 그때까지의 프레임으로 파일을 쓴다
            self.failed += 1  # 끝의 계수에서 "남은 프레임"과 따로 센다
            self._fail(exc)
            return None
        self.processed += 1
        try:
            if self.processed == 1:
                self._width, self._height, self._origin_ms = raw.w, raw.h, status.t_ms
                self._hint = baseline_note(self._session.spec)
                self._say(f"first frame: {self._width}x{self._height}; the session clock starts here")
                self._say(self._hint)
            for event in status.new_events:
                self._say(format_event_line(event, self._origin_ms))
                self._events.append(event_fields(event, self._origin_ms))
            for alert in status.new_alerts:
                self._say(format_alert_line(alert, self._origin_ms))
                self._alerts.append(alert_fields(alert, self._origin_ms))
        except Exception as exc:
            self._fail(exc)
        return raw

    def _offer_preview(self, frame: LiveFrame) -> None:
        """방금 처리한 프레임으로 미리보기를 낸다(`PREVIEW_INTERVAL_S`에 한 번까지). 만들다 오류가 나면 미리보기만 끈다."""
        if not self._preview_on or self._clock() - self._last_preview_s < PREVIEW_INTERVAL_S:
            return
        try:
            self._server.publish_preview(encode_preview(frame.rgb))
        except Exception as exc:  # 미리보기는 세션에 꼭 필요한 것이 아니다
            self._preview_on = False
            self._say(
                f"preview: stopped ({_error_text(exc)}); the session continues without the camera preview"
            )
        self._last_preview_s = self._clock()

    def frames(self) -> Iterator[RawFrame]:
        reader = self._reader
        opened_s = last_line_s = self._clock()
        last_received = last_processed = 0
        hinted = False
        self._publish(PHASE_WAITING)
        while self.reason is None:
            frame = None if self._stop.is_set() else reader.get(POLL_S)
            if self._stop.is_set():  # 기다리는 사이에 왔어도 같다: 새 프레임을 시작하지 않는다
                self.reason = self._stop_reason()
                self._say("stop requested: ending the session")
                break
            if frame is None:
                if reader.finished:
                    self.reason = _end_reason(reader.error)
            else:
                raw = self._process(frame)
                if raw is not None:
                    if self.processed == 1:
                        # 상태 줄의 1초는 첫 프레임부터 센다(그 전의 기다린 시간이 초당 프레임 수에 섞이지 않게)
                        last_line_s, last_received, last_processed = self._clock(), reader.received, 1
                    yield raw  # 파이프라인이 받은 프레임은 빠짐없이 표에 들어간다
                    self._offer_preview(frame)
                    if self.reason is None and self.processed >= MAX_SESSION_FRAMES:
                        self.reason = END_FRAME_LIMIT
            now_s = self._clock()
            if self.processed == 0:
                if not hinted and now_s - opened_s >= FIRST_FRAME_HINT_S:
                    hinted = True
                    self._say(
                        f"no frame yet, {FIRST_FRAME_HINT_S:g} s after the source opened. A stream starts at a key "
                        "frame: if this lasts, give the sender a short key-frame interval (ffmpeg: -g 30)."
                    )
            elif now_s - last_line_s >= STATUS_INTERVAL_S:
                received = reader.received
                self._fps_received = (received - last_received) / (now_s - last_line_s)
                self._fps_processed = (self.processed - last_processed) / (now_s - last_line_s)
                last_line_s, last_received, last_processed = now_s, received, self.processed
                self._say(format_status_line(self.status()))
            self._publish(PHASE_RUNNING if self.processed > 0 else PHASE_WAITING)

        reader.stop()  # 더 읽지 않게 한 뒤에 계수를 읽는다(끝의 수가 서로 맞게)
        self._publish(PHASE_ENDED)
        if self.processed == 0:
            raise LiveError(
                f"the session ended before any frame was processed ({self.reason}); nothing was written"
            )
        counts, session = self._counts, self._session
        # 밀려서 버린 수를 먼저, 디코딩한 수를 나중에 읽는다(읽기 스레드가 아직 돌고 있어도 남은 수가 음수가 되지 않게.
        # 버렸거나 처리했거나 처리하다 실패한 프레임은 모두 그보다 먼저 디코딩된 것이다)
        dropped = reader.dropped
        skipped = counts.skipped + session.skipped
        decoded = counts.decoded
        left = max(0, decoded - skipped - dropped - self.failed - self.processed)
        self._say(f"session ended: {self.reason}")
        self._say(
            f"frames: decoded {decoded}, skipped {skipped} (timestamp), dropped {dropped} (processing was behind), "
            f"left {left} (unprocessed at the end), failed {self.failed} (processing error), "
            f"processed {self.processed}, with a face {session.face_frames}"
        )
        # `extract.write_raw_frames`는 프레임을 다 받은 뒤에 메타데이터를 읽는다. 끝나야 아는 값을 여기서 채운다.
        self._metadata.update(
            {
                "n_decoded": decoded,
                "n_skipped": skipped,
                "n_dropped": dropped,
                "n_left": left,
                "n_failed": self.failed,
                "n_decode_errors": counts.decode_errors,
                "n_processed": self.processed,
                "n_face": session.face_frames,
                "width": self._width,
                "height": self._height,
                "arrival_clock": counts.arrival_clock,
                "end_reason": self.reason,
            }
        )


def _end_reason(error: BaseException | None) -> str:
    """읽기 스레드가 끝난 이유(영어 문장)."""
    if error is None or isinstance(error, EOFError):
        return END_OF_STREAM
    if isinstance(error, av.error.ExitError):
        return END_STALLED
    return f"the source failed ({error})"


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def session_stem(now: datetime | None = None) -> str:
    """출력 파일 이름의 줄기: `live_<YYYYMMDD_HHMMSS>`(세션을 시작한 지역 시각)."""
    if now is None:
        now = datetime.now()
    return f"live_{now:%Y%m%d_%H%M%S}"


def table_metadata(source: str, rotation: int, spec: PipelineSpec, rotation_given: bool = True) -> dict[str, Any]:
    """프레임 표의 메타데이터 가운데 세션을 시작할 때 아는 값. 나머지는 `LiveLoop`가 세션이 끝날 때 채운다.

    키 이름은 추출 결과(`extract.extract_video`)와 같게 둔다. 그래서 `python -m drowsy.demo --raw`가 MediaPipe 모델의
    sha256을 대조하고 회전을 수동 값으로 알린다. `rotation_given`은 `--rotation`을 줬는지다(거짓이면 기본값 0을 썼다.
    모듈 설명의 가정 참조).
    """
    return {
        "source": source,
        "live": True,
        "model_sha256": spec.mediapipe.model_sha256,  # make_landmarker가 파일과 같다고 확인한 값
        "mediapipe_version": extract.mediapipe_version(),
        "drop_sim": None,
        "rotation": rotation,
        "rotation_override": rotation,
        "rotation_given": rotation_given,
    }


def _help_epilog() -> str:
    """`--help`의 끝에 붙는 안내(영어): Windows의 카메라로 돌리는 차례와 보내는 쪽의 명령 예."""
    return "\n".join(
        [
            "with a camera on Windows (this program runs in WSL):",
            f"  1. start this program. It listens ({DEFAULT_SOURCE}) and waits up to {CONNECT_TIMEOUT_S:g} s for the",
            "     sender; a sender that is started first is refused and ends.",
            "  2. while it waits, start the sender in Windows PowerShell (ffmpeg; an example, set the size and the",
            "     frame rate to a mode of your camera):",
            f"       {SENDER_COMMAND.format(target=sender_target(DEFAULT_SOURCE))}",
            f"     camera names:           {SENDER_LIST_DEVICES}",
            f"     sizes and frame rates:  {SENDER_LIST_OPTIONS}",
            "     Keep a short key-frame interval (-g 30 is one second at 30 fps): the session starts at the first",
            "     key frame. Do not add -re or an output -r (they change the frame timing). If the sender is refused",
            "     while this program waits, or the sender has started and this program still prints only",
            f'     "still waiting" lines, try {sender_target(DEFAULT_SOURCE).replace("localhost", "127.0.0.1")}.',
            f"  3. open http://localhost:{DEFAULT_PORT}/ in the browser: the state and the camera preview.",
            "  4. the first 60 s of the session are the baseline: look ahead normally. Until it is complete,",
            "     eye-closure events are provisional and looking away gives no event.",
            "  5. stop with Ctrl+C in this terminal, or stop the sender. The frame table, the timeline PNG and the",
            '     events JSON are written then; keep the window open until the "wrote" lines have appeared.',
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.live",
        description=textwrap.fill(
            "Run the whole pipeline live on a camera stream (the first 60 s of the session are the baseline), show "
            "the state in the terminal and on a local web page, and at the end write the frame table of the "
            "session, a timeline PNG and an events JSON. Face pictures are never written to disk.",
            width=100,
        ),
        epilog=_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=demo.DEFAULT_CONFIG_PATH, help="experiment config YAML")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=(
            "anything PyAV can open. Default: listen on TCP for an MPEG-TS (H.264) stream from the sender "
            f"({DEFAULT_SOURCE}). Also a camera device such as /dev/video0, or a video file (played back at its "
            "own frame timing)"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "exported ONNX model (default: demo.model_path of the config; while that file does not exist, "
            f"export/{demo.FALLBACK_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--speed-kmh",
        type=float,
        default=None,
        help="constant vehicle speed for the alert speed gates (default: not given, alerts are not suppressed)",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=extract.ROTATION_CANDIDATES,
        default=None,
        help=(
            "turn the picture counter-clockwise by this angle to make it upright (default: 0, not turned). A live "
            "stream has no automatic rotation check: look at the preview"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port of the web page on {BIND_HOST} (default {DEFAULT_PORT}; 0 lets the system pick a free port)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="do not make or serve the camera preview (the page shows the state only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            f"output folder (default: <demo.out_dir>/{DEFAULT_OUT_SUBDIR}, i.e. out/{DEFAULT_OUT_SUBDIR}). It is "
            "created and checked for writing before the session starts"
        ),
    )
    return parser


def _remove_dirs(folders: Sequence[Path]) -> None:
    """`_prepare_out_dir`가 만든 폴더를 안쪽부터 지운다. 비어 있지 않은 폴더는 그대로 둔다."""
    for folder in reversed(folders):
        try:
            folder.rmdir()
        except OSError:
            return


def _prepare_out_dir(out_dir: Path) -> list[Path]:
    """출력 폴더를 만들고 그 안에 파일을 만들 수 있는지 확인한다. 새로 만든 폴더들을 바깥쪽부터 돌려준다.

    세션의 표는 끝에 한 번 쓰므로, 쓸 수 없는 폴더는 세션을 시작하기 전에 알아야 한다. 확인은 빈 파일 하나를 만들었다
    지우는 것이다. 만들 수 없거나 쓸 수 없으면 만든 폴더를 되돌리고 `LiveError`다. 아무것도 쓰지 않고 끝나는 실행은
    돌려받은 폴더를 `_remove_dirs`로 지운다(아무것도 남기지 않는다). 예외: 보내는 쪽을 기다리는 동안 SIGTERM이나
    SIGHUP(터미널 창을 닫음)으로 끝난 프로세스는 지우는 코드에 이르지 못해 빈 폴더가 남는다(모듈 설명의 설계 결정 4).
    """
    missing: list[Path] = []
    folder = out_dir
    while not folder.exists() and folder != folder.parent:
        missing.append(folder)
        folder = folder.parent
    created: list[Path] = []
    try:
        for folder in reversed(missing):
            folder.mkdir()
            created.append(folder)
        probe = out_dir / f".live_write_check_{os.getpid()}"
        probe.touch(exist_ok=False)
        probe.unlink()
    except OSError as exc:
        _remove_dirs(created)
        raise LiveError(
            f"the output folder {out_dir} cannot be used ({exc}). Fix: pass --out with a folder that can be created "
            "and written."
        ) from exc
    return created


def main(argv: Sequence[str] | None = None, *, ignore_sigint_at_exit: bool = False) -> int:
    """`python -m drowsy.live`. 종료 코드: 0 성공(세 파일을 썼다), 2 실행할 수 없음(모듈 설명의 설계 결정 6).

    - 세션이 도는 동안 SIGINT, SIGTERM, SIGHUP의 처리기를 표시만 세우는 것으로 바꾸고 끝날 때 되돌린다.
      ignore_sigint_at_exit가 참이면 세션을 돌린 뒤에는 SIGINT를 되돌리지 않고 무시 상태(`SIG_IGN`)로 둔다: 명령으로
      실행할 때(`python -m`) 파일을 다 쓴 뒤 인터프리터가 끝나는 동안의 Ctrl+C가 프로세스를 죽여 종료 코드를 바꾸지 않게
      하려는 것이다(Python 처리기는 인터프리터가 끝날 때 기본 동작으로 되돌려지므로 무시 상태로 바꿔 둔다).
    - 다른 프로그램 안에서 부를 때: 세션이 도는 동안 `gc.freeze()`를 걸었다가 끝날 때 푼다. 보내는 쪽을 기다리다
      중단되면(KeyboardInterrupt) 여는 스레드(`drowsy-live-open`)와 듣는 소켓이 PyAV의 시간 초과(`CONNECT_TIMEOUT_S`)까지
      남고, 읽기에 막힌 채 끝난 읽기 스레드(`drowsy-live-reader`)와 그 컨테이너는 `STALL_TIMEOUT_S`까지 남는다(둘 다 데몬
      스레드다). 그 사이에 같은 프로세스에서 같은 스트림 포트를 다시 열 수 없다. 명령으로 실행하면 프로세스와 함께 끝난다.
    """
    args = _build_parser().parse_args(argv)
    rotation = 0 if args.rotation is None else args.rotation
    rotation_given = args.rotation is not None

    stop = threading.Event()
    signals_seen: list[int] = []
    session: LiveSession | None = None
    server: StatusServer | None = None
    container: Any = None
    reader: LatestFrameReader | None = None
    previous_handlers: dict[int, Any] = {}
    created_dirs: list[Path] = []
    wrote = False
    try:
        # --- 시작: 설정, 출력 폴더, 모델, 랜드마커, 웹 페이지, 소스 (어긋나면 2) ---
        try:
            cfg = load_config(args.config)
            if args.speed_kmh is not None and not (math.isfinite(args.speed_kmh) and args.speed_kmh >= 0.0):
                raise LiveError(f"speed must be a finite value >= 0 km/h, got {args.speed_kmh}")
            if not 0 <= args.port <= 65535:
                raise LiveError(f"--port must be between 0 and 65535, got {args.port}")
            onnx_path = args.model
            if onnx_path is None:
                onnx_path = cfg.demo.model_path
                if not onnx_path.is_file():
                    fallback = cfg.paths.export_dir / demo.FALLBACK_MODEL_NAME
                    print(
                        f"note: demo.model_path {onnx_path} does not exist yet; using the random-weight contract "
                        f"model {fallback}",
                        file=sys.stderr,
                    )
                    onnx_path = fallback
            out_dir = Path(args.out) if args.out is not None else cfg.demo.out_dir / DEFAULT_OUT_SUBDIR
            created_dirs = _prepare_out_dir(out_dir)  # 끝에 가서야 쓸 수 없다는 것을 알면 세션을 잃는다
            feature_spec = load_feature_spec(cfg.paths.feature_spec)
            spec = feature_spec.pipeline
            rules = load_rules(cfg.paths.rules)
            model = demo.verify_model(Path(onnx_path), feature_spec)
            model_fn = demo.make_model_fn(model.path, spec.step_features.window_steps, len(spec.step_features.names))
            landmarker = extract.make_landmarker(cfg.paths.model_path, spec)
            try:
                session = LiveSession(spec, rules, model_fn, args.speed_kmh, landmarker)
            except BaseException:
                landmarker.close()
                raise
            metadata = table_metadata(args.source, rotation, spec, rotation_given)
            try:
                server = StatusServer(args.port, preview=not args.no_preview)
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                raise LiveError(
                    f"port {args.port} on {BIND_HOST} is already in use (another drowsy.live or another program). "
                    "Fix: stop that program or pass --port with a free port."
                ) from exc
            server.start()
            plan = plan_source(args.source)
            described = {"rotation_given": rotation_given, "drowsiness_source": rules.drowsiness.source}
            run_args = (model, args.source, plan, rotation, args.speed_kmh)
            for line in startup_lines(*run_args, server.url, server.preview, CONNECT_TIMEOUT_S, **described):
                _say(line)
            server.publish(
                {
                    "phase": PHASE_WAITING,
                    "preview": server.preview,
                    "source_open": False,
                    "info": run_lines(*run_args, CONNECT_TIMEOUT_S, **described),
                }
            )
            container = open_source(args.source, CONNECT_TIMEOUT_S, _say)
        except _FATAL_ERRORS as exc:
            _complain(str(exc))
            return 2

        # --- 세션 ---
        if plan.listening:
            _say("source opened: the sender is connected and sends video; waiting for the first key frame")
        else:
            _say("source opened: waiting for the first frame")

        def on_signal(signum: int, _frame: Any) -> None:  # 표시만 세운다. 루프가 처리 중인 프레임을 마치고 끝낸다
            signals_seen.append(signum)
            stop.set()

        def stop_reason() -> str:
            signum = signals_seen[0] if signals_seen else signal.SIGINT
            if signum == signal.SIGINT:
                return END_INTERRUPTED
            return END_SIGNAL.format(name=signal.Signals(signum).name)

        for signum in _STOP_SIGNALS:
            try:
                previous_handlers[signum] = signal.signal(signum, on_signal)
            except ValueError:  # 주 스레드가 아니면(테스트) 처리기를 달 수 없다
                break
        counts = StreamCounts()
        reading_stop = threading.Event()  # 읽기를 멈추라는 표시: 읽기 스레드와 프레임 반복자가 함께 본다
        frames = iter_live_frames(container, rotation, counts, stop=reading_stop)
        reader = LatestFrameReader(frames, realtime=plan.realtime, stop=reading_stop)
        page_info = run_lines(*run_args, CONNECT_TIMEOUT_S, opened=True, **described)
        loop = LiveLoop(reader, session, server, counts, stop, metadata, page_info, stop_reason=stop_reason)
        parquet_path = out_dir / f"{session_stem()}.parquet"
        # 시작할 때 만든 객체(라이브러리, 모델)를 세션 도중의 전체 수거가 다시 훑지 않게 한다. 훑으면 프레임 하나의
        # 처리가 그만큼 늦는다(모듈 설명의 전제와 한계에 잰 값이 있다). 세션이 끝나면 푼다.
        try:
            gc.collect()
            gc.freeze()
            reader.start()
            # 표를 쓰는 함수가 세션을 끌고 간다: 프레임을 하나 달라고 할 때마다 루프가 프레임 하나를 처리해 넘긴다.
            extract.write_raw_frames(loop.frames(), parquet_path, spec, metadata)
        except Exception as exc:  # 프레임이 하나도 없었거나(LiveError) 표를 쓰지 못했다
            _complain(_error_text(exc))
            return 2
        finally:
            gc.unfreeze()
            if reader.stop():
                container.close()
            container = None
            session.close()
            session = None
        wrote = True

        # --- 끝: 표에서 데모와 같은 그림과 JSON ---
        _say(f"wrote {parquet_path} ({parquet_path.stat().st_size} bytes)")
        angle = "from --rotation" if rotation_given else "the default; --rotation was not given"
        _say(
            "replay: the lines below are the output of python -m drowsy.demo --raw on that table. Its \"manual "
            f"value recorded by the extraction\" is the rotation of this session: {rotation} ({angle}; not checked)."
        )
        demo_argv = ["--config", str(args.config), "--raw", str(parquet_path), "--model", str(onnx_path)]
        demo_argv += ["--out", str(out_dir)]
        if args.speed_kmh is not None:
            demo_argv += ["--speed-kmh", repr(args.speed_kmh)]
        try:
            code = demo.main(demo_argv)
            sys.stdout.flush()
        except OSError:
            # 터미널이 사라져(SIGHUP 뒤) 데모의 줄을 낼 수 없었다. 데모는 줄을 내기 전에 그림과 JSON을 이미 썼다.
            names = (f"{parquet_path.stem}_timeline.png", f"{parquet_path.stem}_events.json")
            code = 0 if all((out_dir / name).is_file() for name in names) else 2
        if loop.failure is not None:
            _complain(f"the session was ended by a processing error: {_error_text(loop.failure)}")
            return 2
        return code
    except KeyboardInterrupt:  # 소스가 열리기를 기다리는 동안(처리기를 달기 전)의 Ctrl+C
        _complain("interrupted before the session started; nothing was written")
        return 2
    finally:
        if container is not None and (reader is None or reader.stop()):
            container.close()
        if session is not None:
            session.close()
        if server is not None:
            server.stop()
        for signum, previous in previous_handlers.items():
            if ignore_sigint_at_exit and signum == signal.SIGINT:
                signal.signal(signum, signal.SIG_IGN)
            elif previous is not None:
                signal.signal(signum, previous)
        if not wrote:
            _remove_dirs(created_dirs)


if __name__ == "__main__":
    sys.exit(main(ignore_sigint_at_exit=True))
