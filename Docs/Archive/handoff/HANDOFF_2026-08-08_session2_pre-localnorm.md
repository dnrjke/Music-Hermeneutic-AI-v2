# HANDOFF — 2026-08-08 · Q1 rescue 청취 검증 + 천체 기법 응용 탐색

이 문서는 **세션 경계 상태 + 연구 일지**다. 프로젝트 개요는 `README.md`.
v1 프로젝트(`E:\game\Music Hermeneutic AI\`)와는 독립된 후속 프로젝트.

**작업 이력의 전문은 아카이브에 있다** — `Docs/Archive/handoff/`.

---

## 0. 한 문단으로 — 지금 어디인가

**Q1 프로토타입 rescue가 유효 판정을 받았고, 전 트랙 청취 검증이 진행 중이다.**
SuperFlux 포락선 → Otsu 피크 → Q1 rescue → 4초 윈도우 계수 → 블라인드 설문.
Q1전체_클릭이 가장 안정적인 음원으로 판정. morphological opening 접근은
하드코어 음악에서 무효(거의 모든 사건이 충격적, filtered=0) — 스템 제거됨.
별-성운 분리 기법(분광 균일도 등)의 음향 응용을 탐색 중.

---

## 1. 세션 이력

### 세션 1 (2026-08-07~08) — 파이프라인 구축 + Q1 발견

v1 코드 이식 → 파이프라인 → 맛보기 4응답 → 피아노 과소탐지 진단 → 6종 보정 시도
→ Q1 프로토타입 rescue 유효 확인 (GL 0-4s: 12→19, 인간 18).
mid-band rescue 오탐 판정. HANDOFF 생성, GitHub push 완료.

### 세션 2 (2026-08-08, 현재) — 청취 검증 + opening 시도/실패 + 천체 응용

#### 2-a. morphological opening 시도 — 실패

성운 처리의 별-성운 분리(grey_opening)를 음향에 적용:
- opening 단독: GL 16-20s 6→4 (인간 4 일치) — 점진 사건 제거 효과
- opening 단독: GL 0-4s 12→9 — 피아노도 충격이라 오히려 악화
- opening → Q1 조합: VN 264→718 폭증
- Q1 → opening 필터: **filtered=0** — 하드코어에서 모든 사건이 충격적

**결론: opening은 하드코어 음악에 무효.** 접근 자체가 장르에 부적합.

#### 2-b. 전 트랙 소니파이 생성 — 청취 검증

`out/sonify/{트랙}/{구간}/` 구조로 재정리:
```
원곡.wav           — 원본
1차탐지_클릭.wav    — Otsu 초과만 (3kHz)
Q1전체_클릭.wav     — 1차(3kHz) + 구조(5kHz)
Q1구조만_클릭.wav   — 구조 피크만 (5kHz)
```

#### 2-c. 사용자 청취 관측

- **Q1전체_클릭이 가장 안정적인 음원** (사용자 판정)
- **cry 0-4s 과다탐지**: 조용한 구간에서 base 3→Q1 18. 프로토타입 불안정 가능성.
- **cry 링잉 이중 탐지**: 링잉(잔향)에서 2회 찍히는 현상 관측 (급한 이슈 아님).
- **GL 16-20s 신스 킥 4회**: 사용자 인지 4회를 어떤 방법도 정확히 캐치 못함.
- **연속 신스 비트**: Q1이 많이 찍은 구간에서 연속 신스 비트를 잘 잡아냄 (긍정 평가).
- opening 방식 스템은 전량 삭제됨.

#### 2-d. 천체 기법 응용 탐색 — 진행 중

별-성운 분리의 핵심 원리:
- **분광 균일도**: `ch_min/ch_max` — 별=광대역(백색), 성운=협대역(유색)
- **다각도 라인 opening**: 점광원 vs 방향성 필라멘트
- **밴드패스 추출**: 특정 주파수 대역 구조 분리

**"별을 노이즈로 여기고 특정"**하는 발상의 음향 응용을 Fable에게 의뢰.
참조: `E:\game\2Test1\Docs\references\stellar\fits_work\veil_nebula\baseline\BASELINE.md`

---

## 2. **다음 세션은 여기부터 읽는다**

### 현재 상태

- **peak_pick.py**: `peaks_with_mid()` 함수가 pipeline에서 사용 중 — **교체 필요** (오탐 판정)
- **Q1 프로토타입 rescue**: 진단 스크립트에만 존재. pipeline 미통합.
- **설문 클립**: peaks_with_mid 기준 → Q1 적용 시 재생성 필요.
- **소니파이**: `out/sonify/{GL,cry,VN,SS}/{구간}/` — opening 스템 삭제 완료.
- **천체 응용**: Fable 진단 스크립트 결과 대기 중 (`_diag_spectral_identity.py` 예정).

### 다음 단계

1. **Fable 천체 응용 결과 확인** — 새 접근법 유효 시 통합 검토
2. **최종 탐지 방법 확정** → peak_pick.py 통합
3. **파이프라인 재실행** → 설문 재생성
4. **48개 블라인드 설문 응답 수집**
5. **survey_analyse.py 실행**

### 열려 있는 문제

- Q1 rescue pipeline 미통합
- peaks_with_mid 교체 필요
- cry 0-4s 과다탐지 (조용한 구간 프로토타입 불안정)
- cry 링잉 이중 탐지
- GL 16-20s 신스 킥 4회 미캐치
- survey_gen.py append 문제 (재실행 시 ground_truth.json 수동 삭제)
- 분류 모듈(classify.py) — 총 계수 검증 통과 후

---

## 3. 파이프라인

```
오디오 → LUFS 정규화 → STFT → mel → SuperFlux 온셋 포락선
                                          ↓
                              Otsu 임계 → 극대점 → 30ms 최소간격 → 이산 사건
                                                                      ↓
                                              (Q1 프로토타입 rescue) → 4초 윈도우 계수
                                                                      ↓
                                                              블라인드 설문
```

### Q1 프로토타입 rescue 원리

1. 1차 탐지 피크(Otsu 초과)의 대역 프로파일 [low, mid, high] 추출
2. 프로파일 중앙값 = 원형(prototype)
3. 탐지 피크 ↔ 원형 간 코사인 유사도의 Q1(25%) = 구조 하한
4. Otsu 미달 극대점 중 원형 유사도 ≥ Q1인 것을 구조
5. 전체 후보 진폭 내림차순 탐욕 선택 + 30ms 최소간격

**자유 파라미터 0개** — D-21 준수.

---

## 4. 코드 변경 상태

| 파일 | 상태 | 비고 |
|------|------|------|
| `src/config.py` | 안정 | 상수, 경로 |
| `src/audio_io.py` | 안정 | v1 이식 |
| `src/onset.py` | 안정 | SuperFlux + 3밴드 |
| `src/peak_pick.py` | **교체 필요** | peaks_with_mid → Q1 rescue |
| `src/counter.py` | 안정 | |
| `src/survey_gen.py` | 안정 | append 버그 주의 |
| `src/null_model.py` | 안정 | |
| `src/survey_analyse.py` | 미사용 | 응답 대기 |
| `src/pipeline.py` | **교체 필요** | peaks_with_mid 사용 중 |
| `src/sonify.py` | 안정 | 청각 검증 |

### 진단 스크립트

| 파일 | 판정 |
|------|------|
| `_diag_piano.py` | 원인 분석 완료 |
| `_diag_union*.py` | 과대 — 기각 |
| `_diag_midrescue*.py` | 오탐 — 기각 |
| `_diag_local_otsu.py` | VN 폭발 — 기각 |
| `_diag_prototype.py` | 전역 프로토타입 — 기각 |
| `_diag_proto_local.py` | min 기준 — 과대 |
| `_diag_proto_q1.py` | **Q1 기준 — 유효** |
| `_diag_morpho.py` | opening 단독 — 하드코어 무효 |
| `_diag_morpho_q1.py` | opening+Q1 — VN 폭증 |
| `_diag_q1_then_filter.py` | Q1→opening 필터 — filtered=0 |
| `_sonify_proto.py` | GL 전용 소니파이 |
| `_sonify_q1_all.py` | 전 트랙 소니파이 (현행) |
| `_sonify_opening.py` | opening 소니파이 (스템 삭제됨) |
| `_diag_spectral_identity.py` | **대기 중** — Fable 천체 응용 |

---

## 5. 규율

| ID | 규율 |
|---|---|
| D-07 | 에너지 계열 금지 — SuperFlux(스펙트럼 변화) 사용 |
| D-18 | 천장(r~0.85)과 바닥(r~0) 사전 정의 |
| D-21 | 출력 보고 파라미터 고르지 않는다 — Otsu(0 파라미터), Q1(데이터 도출) |
| D-24 | 관측과 해석을 분리 |
| D-v2-01 | 설문 완료 전 ground_truth.json 열지 않음 |
| D-v2-03 | 윈도우 4초 고정 |

---

## 6. 환경

```
Python 3.14.2  (C:\Python314)
.venv          v1과 공유 (E:\game\Music Hermeneutic AI\.venv)
               torch/demucs 불필요 — librosa/numpy/soundfile/soxr/pyloudnorm만
GPU            RTX 3080 10 GB
```

콘솔 cp932 — 한글 출력은 `reconfigure(encoding="utf-8")`.
오디오: v2/audio/ 우선, v1 `audio/target/` 폴백.
대상 4곡: cry of viyella, Grievous Lady, Viyella's Nightmare, Swift Swing.

---

## 7. 맛보기 응답 (참고용)

| 클립 | 구간 | 시스템 | 인간 | Δ |
|------|------|--------|------|---|
| GL 0-4s | 피아노 | 12 | 18 | -6 |
| GL 16-20s | 신스 기음 | 6 | 4 | +2 |
| VN 200-204s | 목소리 | 2 | 6 | -4 |
| VN 268-272s | 신스 킥 | 12 | 14 | -2 |

Q1 rescue 적용 시: GL 0-4s → 19, VN 200-204s → 6.
