# HANDOFF — 2026-08-08 · 사건 탐지 파이프라인 · Q1 프로토타입 rescue 유효 확인

이 문서는 **세션 경계 상태**다. 프로젝트 개요는 `README.md`.
v1 프로젝트(`E:\game\Music Hermeneutic AI\`)와는 독립된 후속 프로젝트.

**작업 이력의 전문은 아카이브에 있다** — `Docs/Archive/handoff/`.

---

## 0. 한 문단으로 — 지금 어디인가

**이산 사건 탐지 + 블라인드 계수 검증 파이프라인이 가동 중이다.**
SuperFlux 포락선 → Otsu 피크 → 4초 윈도우 계수 → 블라인드 설문.
맛보기 4개 응답에서 피아노 건반음 과소탐지(시스템 12 vs 인간 18)를 발견.
6가지 보정 시도 끝에 **Q1 프로토타입 rescue**가 유효 — 1차 탐지 피크의
국소 대역 프로파일을 원형 삼아 유사도 상위 75%인 미달 극대점을 구조.
GL 0-4s: 12→19 (인간 18). 사용자 청취 검증에서 **상당히 유효하다고 판정.**
mid-band rescue는 오탐 판정으로 기각됨.

---

## 1. 이번 세션에서 일어난 일

### 1-a. v2 파이프라인 구현 — 완료

v1의 SuperFlux/Otsu/HPSS 코드를 이식하여 경량 파이프라인 구축.
torch/demucs 불필요, librosa/numpy/soundfile만 사용.

```
오디오 → LUFS 정규화 → STFT → mel → SuperFlux 온셋 포락선
                                          ↓
                              Otsu 임계 → 극대점 → 30ms 최소간격 → 이산 사건
                                                                      ↓
                                                   4초 윈도우별 총 계수 → 블라인드 설문
```

4곡 48개 설문 클립 생성, 정답 봉인(`ground_truth.json`).

### 1-b. 맛보기 4개 응답 수집

| 클립 | 구간 | 시스템 | 인간 | Δ | 묘사 |
|------|------|--------|------|---|------|
| 0fee555c | GL 0-4s | 12 | 18 | -6 | 피아노 건반음 |
| 0d82dd83 | GL 16-20s | 6 | 4 | +2 | 신스 기음 |
| 102a9fee | VN 200-204s | 2 | 6 | -4 | 목소리 덩어리 |
| 12454d36 | VN 268-272s | 12 | 14 | -2 | 신스 킥 |

**핵심 문제**: 피아노 건반음 과소탐지 (12 vs 18).

### 1-c. 피아노 과소탐지 진단

전역 Otsu 임계(1.23)가 트랙 몸통의 퍼커시브 구간에 의해 높게 설정됨.
피아노 인트로의 온셋값(0.3~1.0)이 이 임계 아래로 잘린다.

진단 데이터:
- GL 전역 Otsu: 1.2292, 피아노 구간(0-4s) 극대점 135개 중 12개만 통과
- mid-band(120-2000Hz) 단독: 17개 탐지 (거의 일치!)
- 구간 local Otsu: 0.7946 → 13개 (미미한 개선)

### 1-d. 보정 시도 6종 — 결과 요약

| # | 방법 | GL 0-4s | 전체 변화 | 판정 |
|---|------|---------|-----------|------|
| 1 | 3대역 합집합(L+M+H) | 27 | +114% | 과대 |
| 2 | full+mid 병합 | 17 | VN +166% | VN 폭발 |
| 3 | full+3대역 rescue (극대점 제약) | 21 | +65% | 과대 |
| 4 | Local Otsu (8/16/32s) | 13 | VN +485% | GL 무효, VN 폭발 |
| 5 | mid-band만 rescue (극대점 제약) | 16 | +11% | 부족 |
| 6 | **Q1 프로토타입 rescue** | **19** | +8% | **유효** |

full+mid 병합을 peak_pick.py에 적용 후 파이프라인 재실행했으나,
**사용자 청취 검증에서 mid-rescued 피크는 오탐 판정** → mid-merge 기각.

### 1-e. Q1 프로토타입 rescue — 유효 확인

**원리**: 각 4초 윈도우에서—
1. 1차 탐지 피크(Otsu 초과)의 대역 프로파일 [low, mid, high] 추출
2. 프로파일 중앙값 = 원형(prototype)
3. 탐지 피크 ↔ 원형 간 코사인 유사도의 Q1(25번째 백분위) = 구조 하한
4. Otsu 미달 극대점 중 원형 유사도 ≥ Q1인 것을 구조
5. 전체 후보 진폭 내림차순 탐욕 선택 + 30ms 최소간격

**자유 파라미터 0개** — D-21 준수:
- 원형: 데이터 자체의 중앙값
- Q1: 탐지 피크 분포에서 도출
- 대역 경계: v1에서 정의 (120/2000 Hz)

GL 0-4s 결과: 1차 12 + 구조 7 = **19** (인간 18, Δ+1).
**사용자가 `_q1_rescued_only.wav`로 구조된 7개를 청취, "상당히 유효"로 판정.**

### 1-f. 청각 검증 도구 (sonify.py) 구현

`src/sonify.py` — 탐지 사건의 청각 검증 파일 생성:
- `*_click.wav`: 원곡 + 탐지 시각마다 3kHz 클릭
- `*_stem.wav`: 탐지 시각 80ms만 추출 (나머지 무음)
- `*_mix.wav`: 스테레오 L=원곡, R=클릭만
- `*_mid_rescued.wav`: mid-band 구조 피크만 5kHz 표시

### 코드 변경

| 파일 | 무엇 | 상태 |
|------|------|------|
| `src/config.py` | 상수, 경로 | 안정 |
| `src/audio_io.py` | 오디오 적재 (v1 이식) | 안정 |
| `src/onset.py` | SuperFlux 포락선 (전대역 + 3밴드) | 안정 |
| `src/peak_pick.py` | Otsu 피크 + peaks_with_mid (현재 적용됨) | 재검토 필요 |
| `src/counter.py` | 4초 윈도우 계수 | 안정 |
| `src/survey_gen.py` | 블라인드 설문 클립/정답 생성 | 안정 |
| `src/null_model.py` | 무작위 배치 귀무 모형 | 안정 |
| `src/survey_analyse.py` | 시스템 vs 인간 비교 | 미사용 (응답 대기) |
| `src/pipeline.py` | 전체 파이프라인 (현재 peaks_with_mid 사용) | 재검토 필요 |
| `src/sonify.py` | 탐지 사건 청각 검증 | 안정 |

### 진단 스크립트 (일회성)

| 파일 | 무엇 |
|------|------|
| `src/_diag_piano.py` | 피아노 과소탐지 원인 분석 |
| `src/_diag_union.py` | full+3대역 합집합 시도 |
| `src/_diag_union2.py` | 3대역/full+mid/대역별 비교 |
| `src/_diag_union3.py` | rescue + bandmax-norm 비교 |
| `src/_diag_midrescue.py` | mid 전용 rescue (극대점 제약) |
| `src/_diag_midrescue2.py` | mid 전용 병합 (극대점 제약 없음) |
| `src/_diag_local_otsu.py` | 슬라이딩 윈도우 local Otsu |
| `src/_diag_prototype.py` | 전역 프로토타입 rescue (실패) |
| `src/_diag_proto_local.py` | 국소 프로토타입 rescue (min 기준, 과대) |
| `src/_diag_proto_q1.py` | 국소 프로토타입 Q1 기준 비교 |
| `src/_sonify_proto.py` | Q1 rescue 청각 검증 파일 생성 |

---

## 2. **다음 세션은 여기부터 읽는다**

### 현재 상태

- **peak_pick.py**: `peaks_with_mid()` 함수가 추가되어 있고 pipeline.py에서 사용 중.
  **그러나 사용자가 mid_rescued를 오탐으로 판정** — peaks_with_mid는 더 이상 유효하지 않음.
- **Q1 프로토타입 rescue**: 진단 스크립트(`_diag_proto_q1.py`, `_diag_proto_local.py`)에만 존재.
  아직 peak_pick.py / pipeline.py에 통합되지 않음.
- **설문 클립**: 현재 peaks_with_mid 기준으로 생성됨 → Q1 적용 시 재생성 필요.

### 다음 단계

1. **Q1 프로토타입 rescue를 peak_pick.py에 통합** — peaks_with_mid 대체
2. **파이프라인 재실행** → 설문 재생성
3. **사용자의 추가 방향** — "이어서 할 말 있음"으로 대기 중

### 사용자 결정사항 대기

- 사용자가 "이어서 할 말 있음"으로 추가 지시 예정
- "특정한 형태가 반복되는 경우 묶을 수 있는지" — 대역별 탐지를 렌즈로 해석하는 방식 탐색 의사

### 규율

| ID | 규율 |
|---|---|
| D-07 | 에너지 계열 금지 — SuperFlux(스펙트럼 변화) 사용 |
| D-18 | 천장(r~0.85)과 바닥(r~0) 사전 정의 |
| D-21 | 출력 보고 파라미터 고르지 않는다 — Otsu(0 파라미터), Q1(데이터 도출) |
| D-v2-01 | 설문 완료 전 ground_truth.json 열지 않음 |
| D-v2-03 | 윈도우 4초 고정 |

---

## 3. 환경

```
Python 3.14.2  (C:\Python314)
.venv          v1과 공유 (E:\game\Music Hermeneutic AI\.venv)
               torch/demucs 불필요 — librosa/numpy/soundfile/soxr/pyloudnorm만
GPU            RTX 3080 10 GB
```

**콘솔이 cp932** — 한글 출력은 `reconfigure(encoding="utf-8")`.

오디오 파일: v2/audio/ 우선, v1 `audio/target/` 폴백.
대상 4곡: cry of viyella, Grievous Lady, Viyella's Nightmare, Swift Swing.

---

## 4. 열려 있는 문제

- **Q1 rescue 통합**: 진단에서 유효 확인, pipeline 통합 미완
- **peaks_with_mid 제거**: 오탐 판정, pipeline에서 교체 필요
- **설문 48개 응답**: 미수집. Q1 통합 후 재생성 예정
- **분류 모듈**: classify.py — 총 계수 검증 통과 후 착수
- **survey_gen.py append 문제**: ground_truth.json에 기존 데이터 append (재실행 시 삭제 필요)
