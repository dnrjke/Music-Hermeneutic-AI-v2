# s4 piano stem event sculpt

BS-Roformer piano stem을 **전처리 → 소니파이 청취 → 점진 축소**로
이산 사건 잔여에 가깝게 남기는 실험 작업공간이다.

기존 `s4_piano` onset 검출기 경로(A-2 + positive rescue 395 등)와
`stem_validation`/`transcription`과는 분리한다. 395는 비교 기준선으로만 둔다.

## 목표

- 귀납에 가깝게: 검출기 파라미터 고르기보다, 스템을 듣고 남길 사건만 좁혀 간다.
- 누락을 우선 적게: 한 번에 최종을 만들지 않고 여러 패스로 점진 축소한다.
- 산출은 기존 395보다 **청취가 깨끗한 소니파이**(이산 신호에 가까운 잔여).

## 입력 (고정 후보)

- 주 입력: `out/stems/Dir/bs_roformer/piano.wav`
- 비교 기준선: A-2 + positive-distribution rescue 395
  (`out/sonify/Dir/전체_a2_posdist_rescue_클릭.wav` 등)

정답·튜닝 목표로 stem/전사를 쓰지 않는다 (D-21).

## Workspace

`runtime/`, `work/`는 이 디렉터리 아래에 격리하고 git에서 제외한다.
기존 공유 venv와 `stem_validation` runtime은 수정하지 않는다.

## 상태

작업영역만 준비됨. 전처리·소니파이 계획 및 구현은 후속 세션.
