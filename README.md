# EXAONE coding agent

Python 3.10 이상과 Linux의 `bubblewrap`(`bwrap`)이 필요하다. Python 패키지 설치는 필요 없다.
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
2. 지원 클라이언트는 `--reasoning-budget`과 `--final-budget`을 독립 필드로 받는다.
   EXAONE은 입력 하네스에서 독립 추론 상한의 지원이 확인되지 않았다.
   지원하지 않는 필드를 임의로 보내지 않고 총량은 두 예산의 합, effort는 low로 요청한다.
   보고서의 `independent_caps_supported=false`가 이 한계를 명시한다.
   실제 EXAONE 독립 상한 지원 여부와 effort 동작 확인은 후속 실실행 과제다.
3. 전송 재시도는 모델 대화 밖에서 처리한다. 요청/시도 ID·시각·상태·대기를 따로 기록한다.
   429·5xx·네트워크 오류만 재시도하고 숫자/HTTP 날짜 `Retry-After`를 지킨다.
   전체 시간 예산을 넘는 대기는 시작하지 않는다. 개별 timeout도 잔여 예산 이내다.
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

실제 EXAONE API는 호출하지 않았다. 종료 판정, 예산 분리, 429/503 재시도와 시간 예산,
동시 호출 제한, 시간 분리, 실제 sandbox 탈출 거부, 중복 차단을 스텁으로 검사한다.
실행 중인 벤치가 끝난 뒤 실제 제공자 호환성과 품질을 확인해야 한다.
