# EXAONE coding agent

Python 3.10 이상과 Linux의 `bubblewrap`(`bwrap`)이 필요하다. Python 패키지 설치는 필요 없다.
이미 격리된 컨테이너 안에서는 `--no-bwrap`로 bwrap 셸 격리를 건너뛰고 작업 폴더 안에서 직접 실행한다.
기존 벤치 하네스에서 독립된 실행기이며, `guide.txt`는 `guide-weak-v3.txt`의 복사본이다.

## 실행

작업 지시를 텍스트 파일에 쓰고 별도의 작업 폴더를 준비한다.

```sh
export EXAONE_API_KEY='실제 키는 환경변수로만 설정'
python3 agent.py --task task.txt --workdir ./my-task --model exa
```

모델은 기본값이 `exa`다. 정상 종료일 때만 stdout에 최종 답을 쓰고 exit 0으로 끝난다.
실패하면 stdout에는 답을 내지 않고 stderr와 JSON 보고서에 실패 이유를 남기며 exit 1로 끝난다.
보고서는 기본적으로 `logs/` 아래에 생성된다. `--output result.json`으로 위치를 지정할 수 있다.
보고서에는 작업 내용과 도구 출력이 들어가므로 커밋하지 않는다.

API 없이 전체 검증:

```sh
python3 tests.py
```

독립적인 스텁 CLI 실행은 OpenAI 형식의 응답 배열 JSON 파일로 재생한다.
`{"error":{"status":429,"retry_after":"0"}}` 항목도 넣을 수 있다.
테스트의 `cli_end_to_end`가 작은 덧셈 수정 작업, 실제 Python 검사, 중복 검사 차단과 최종 답까지 실행한다.

```sh
python3 agent.py --task task.txt --workdir ./my-task \
  --model stub --replay replay.json --output result.json
```

## 실행기가 강제하는 여섯 장치

1. 요청별 종료 이유·usage·역할·도구 유무를 보존한다. `length`, 빈 답, 잘못된 역할,
   턴 제한은 실패이며 `final`은 null이다. 잘린 도구 호출은 실행하지 않는다.
   최종 답은 도구 없는 정상 assistant `stop` 응답에서만 나온다.
   `<think>` 또는 `</think>`가 최종 본문에 섞여도 실패 처리한다.
2. EXAONE 응답은 SSE 스트림으로 받으며 실행기가 두 예산을 독립적으로 검사한다.
   `reasoning_content`가 실행 전체의 `--reasoning-budget`을 초과하면 즉시 HTTP를 닫고
   `reasoning_budget_exceeded` 실패를 기록한다. `content`가 요청별 `--final-budget`을
   초과하면 `final_budget_exceeded` 실패다. 두 경우 모두 final은 null이고 도구도 실행하지 않는다.
   provider tokenizer가 없어 UTF-8 바이트 하나를 예산 단위 하나로 보수적으로 센다.
   따라서 한글·다중 바이트 문자 등은 실제 tokenizer보다 일찍 중단될 수 있다.
   보고서의 `counting=utf8_byte_upper_bound`와 `budget_usage`가 실제 과금 usage와 이를 구분한다.
   스트림 델타 단위로 검사하므로 초과를 드러낸 델타는 수신되지만 답이나 도구에 반영하지 않는다.
   추론 소비는 턴 및 부분 스트림 전송 재시도 전체에 누적되며 환급되지 않는다.
   예산을 정확히 소진한 뒤 다음 요청은 `enable_thinking=false`, `max_tokens=final_tokens`다.
   제공자가 thinking을 무시해 다시 추론을 보내면 즉시 같은 실패로 중단한다.
   아직 추론 여유가 있으면 서버 max_tokens는 잔여 추론과 답변 예산의 합이지만,
   로컬 스트림 검사가 각각의 상한을 강제한다. 도구 호출 델타는 조립 후 정상 완료일 때만 실행한다.
   `independent_caps_supported=false`는 서버 독립 필드 미지원이고,
   `stream_caps_enforced=true`는 실행기 강제가 활성화되었다는 뜻이다.
   독립 필드를 지원하는 비스트리밍 클라이언트도 기존 인터페이스로 사용할 수 있다.
3. 전송 재시도는 모델 대화 밖에서 처리한다. 요청/시도 ID·시각·상태·대기를 따로 기록한다.
   429·5xx·네트워크 오류만 재시도하고 숫자/HTTP 날짜 `Retry-After`를 지킨다.
   전체 시간 예산을 넘는 대기는 시작하지 않는다. 개별 timeout도 잔여 예산 이내다.
   SSE 파싱 전에 모든 원시 줄(heartbeat·주석·빈 줄 포함)의 읽기 전후에
   같은 절대 마감 시간을 검사하고, 소켓 읽기 timeout을 매번 잔여 시간으로 줄인다.
   요청을 열기 전에 타이머를 시작하고 urllib 연결 생성 단계에서 소켓을 등록한다.
   DNS는 daemon 스레드에서 해석하고 호출자는 큐 결과를 남은 시간만 기다린다.
   마감이 지나면 OS resolver의 종료를 기다리지 않고 `transport_timeout`으로 반환한다.
   늦은 결과는 큐에만 저장되며 소켓을 생성하거나 연결하지 않는다.
   DNS·연결·헤더·본문은 같은 절대 마감을 쓰며 주소 재시도나 응답 수신 후에도 재설정하지 않는다.
   마감 타이머가 등록된 소켓을 shutdown/close하므로 느린 헤더나 줄바꿈 없는 본문도 중단한다.
   실제 연결은 프록시 환경변수를 사용하지 않는 전용 HTTPS opener로 수행한다.
   무응답·마감 초과는 전송 기록의 `error=transport_timeout`으로 남기고
   사용한 시간과 재시도 대기를 동일한 전체 예산에서 차감한다.
   `--retry-budget`(기본 120초), `--request-timeout`(60초), 최대 6회,
   `--concurrency`(1)의 공유 semaphore를 적용한다. 한 CLI의 모델 호출과 도구는 순차다.
   동시성 제한은 같은 executor를 공유하는 호출에 적용되며 여러 CLI 프로세스 사이 전역 제한은 아니다.
4. 실제 전송 시도 합계 `model_seconds`, 도구 합계 `tool_seconds`,
   재시도 대기 `retry_wait_seconds`, 전체 `wall_seconds`를 분리한다.
   모델 시간은 서버 내부 순수 추론 시간이 아니라 HTTP 왕복 시간이다.
5. 읽기·검색·쓰기의 경로를 resolve한 뒤 작업 폴더 경계를 검사한다.
   shell은 host `/`를 마운트하지 않는 bubblewrap에서 실행한다.
   `/usr`의 실행기·라이브러리만 읽기 전용으로 제공하고 작업 폴더만 `/work`로 연결한다.
   홈·부모 벤치·호스트 `/tmp`·환경변수 키와 네트워크는 보이지 않는다.
   `/proc`과 `/dev`는 격리된 실행 환경용이다. 도구의 경로는 상대 경로나 `/work/...`다.
   sandbox가 없거나 동작하지 않아도 host shell로 우회하지 않는다.
   벤치 보호 경로와 겹치는 workdir 및 보고서 출력 경로도 거부한다.
6. 모든 shell 명령에 명령 문자열과 작업 파일 상태 해시를 적용한다.
   내용·경로·권한·symlink 대상이 같으면 재실행 대신 `UNCHANGED`와 이전 결과 요약을 준다.
   실패한 검사도 중복 실행하지 않는다. 파일 변경 후에는 재검사가 가능하다.
   `.git`, `__pycache__`, `.pytest_cache`, `.mypy_cache`는 상태에서 제외한다.
   나머지 작업 파일 전체를 관련 상태로 취급하므로 출력 파일도 변경이면 재실행을 허용한다.
   상태는 실행 전 해시이며 검사가 직접 파일을 바꾸면 새 상태의 검사로 간주하지 않는다.

## 현재 검증 범위

실제 EXAONE API는 호출하지 않았다. 실제 ExaClient의 HTTP를 SSE 재생으로 대체해
초과 시 조기 연결 종료를 검증한다. 스트리밍 추론/답변 초과, 턴 간 누적 및 thinking 해제,
부분 스트림 재시도 후 소비 보존, 종료 판정, 429/503 재시도와 시간 예산,
동시 호출 제한, 시간 분리, 실제 sandbox 탈출 거부, 중복 차단을 스텁으로 검사한다.
느린 헤더·무응답·부분 본문·heartbeat는 localhost 서버와 실제 urllib/socket으로도 검사한다.
이 로컬 검사는 TLS만 우회하며 외부 DNS나 EXAONE에 접속하지 않는다.
별도 주입형 검사는 DNS가 계속 막힌 상태에서도 deadline에 반환하는 것과
늦은 해석 결과의 연결 차단, DNS 오류의 전송 오류 분류를 확인한다.
실행 중인 벤치가 끝난 뒤 실제 제공자 호환성과 품질을 확인해야 한다.
