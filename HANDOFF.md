# HANDOFF — 2026-08-08 · block-gated adaptive 확립 + s4 피아노 진행중

이 문서는 **세션 경계 상태 + 연구 일지**다. 프로젝트 개요는 `README.md`.
v1 프로젝트(`E:\game\Music Hermeneutic AI\`)와는 독립된 후속 프로젝트.

**작업 이력의 전문은 아카이브에 있다** — `Docs/Archive/handoff/`.

---

## 0. 한 문단으로 — 지금 어디인가

**본선 4곡은 block-gated adaptive detection으로 수렴, pipeline 통합 대기.**
별도로 s4_piano 실험 진행중: Dir 피아노곡 대상 onset 탐지.
SuperFlux가 피아노 타건을 구조적으로 놓치는 문제 확인 —
spectral novelty + local norm(2s)이 현재 최선(613 peaks).
등간격 비트 artifact는 저역/배경음 유래가 아닌 **곡의 리듬 격자에 대한
SuperFlux의 균일 반응**(128ms ≈ 16분음표)으로 진단 완료.
**v2 본선(EDM/하드코어)에서는 문제 아님** — 피아노 고유 문제.

---

## 1. 다음 세션은 여기부터 읽는다

### 본선 파이프라인 상태

- **gate+norm잔여 = 최종 adaptive 기법**: SS≈Otsu 유지, VN≈norm 자동 적응
- **소니파이**: `out/sonify/{트랙}/` — 세션 2 6종 + 세션 3 gate 3종
- **pipeline 통합 착수 단계**

### 본선 다음 단계

1. **사용자 gate 소니파이 청취 판정**
2. **경계 이슈 대응 결정** (VN 3:12 블록 0.004 차이)
3. pipeline 통합 → 설문 재생성 → 48개 응답 수집

### 본선 열려 있는 문제

- VN [3:12] 블록 경계 이슈 (99pct ≈ Otsu, 4개 윈도우 과소탐지)
- cry Q1 과다탐지 (SIR 억제 필요)
- cry 링잉 이중 탐지 (matched filtering 후보)
- GL 16-20s 신스 킥 4회 미캐치
- peaks_with_mid 교체 필요 (pipeline.py)
- survey_gen.py append 버그
- 분류 모듈(classify.py) — 총 계수 검증 통과 후

### s4_piano 상태

> **기술 자문 보고서**: [`Docs/s4_piano_advisory.md`](Docs/s4_piano_advisory.md)
> Fable 자문 전문 + 두 방향 정리 + 성운 프로젝트 대응표. 피아노 외에도 참고 자산.

**목표**: `102 - Dir.wav` (dai, 오르골/피아노, ~117s) — 피아노 건반 타건 전수 탐지.

**현재 최선**: novelty + local norm (2s) = 613 peaks (baseline과 공통 13개).
사용자 청취: "마음에 들었으나 미흡."

**s4 다음 단계** (Fable 자문 기반 우선순위):

1. **B-2: CQT 기반 novelty** — mel→CQT 전환. 피아노 배음 분리 개선. 근본 원인.
2. **A-1/A-3: 다중 스케일 정합(WTMM) + 연속값 기하평균 융합** — novelty_norm 점진 개선.
3. **A-2: 슬라이딩 percentile_filter** — 블록 경계 아티팩트 제거.
4. **B-3: Complex-domain flux** — 크기 가중 복소 거리 (위상 단독 실패와 다름).
5. 대역 재정의 후 B-1(교차대역 동시성) 재시도.
6. MuseScore 팬 악보 활용 검토 (참조 정답용, D-21 비저촉).

### 소니파이 파일 현황

```
out/sonify/{트랙}/            ← 본선 4곡
  # 세션 2
  전체_1차탐지_클릭.wav    — Otsu base (3kHz)
  전체_Q1_클릭.wav         — Q1 (base 3kHz + 구조 5kHz)
  전체_SIR_클릭.wav        — SIR(u3) (base 3kHz + 구조 5kHz)
  전체_norm_클릭.wav       — norm(16s) base (3kHz)
  전체_norm+Q1_클릭.wav    — norm + Q1 rescue (3kHz/5kHz)
  전체_norm+SIR_클릭.wav   — norm + SIR rescue (3kHz/5kHz)
  # 세션 3
  전체_gate_클릭.wav       — block-gate (3kHz)
  전체_gate+Q1_클릭.wav    — gate + Q1 rescue (3kHz/5kHz)
  전체_gate+norm잔여_클릭.wav — gate base + norm에만 있는 피크 (3kHz/5kHz)

out/sonify/Dir/              ← s4 피아노 실험
  전체_baseline_클릭.wav        618  SuperFlux Otsu
  전체_plain_클릭.wav           684  plain flux
  전체_nomaxf_클릭.wav          690  no-maxfilter
  전체_novelty_클릭.wav        1052  spectral novelty (raw)
  전체_nov_n2s_클릭.wav         613  ★ novelty + norm 2s (현재 최선)
  전체_nov_n2s_비교_클릭.wav         3kHz=baseline 공통(13), 5kHz=신규(600)
  전체_log1p_raw_클릭.wav       696  log1p (raw)
  ... 외 다수 (블록크기×deburst 조합, intersection, 비교)
```

---

## 2. 세션 이력

### 세션 1 (2026-08-07~08) — 파이프라인 구축 + Q1 발견

v1 코드 이식 → 파이프라인 → 맛보기 4응답 → 피아노 과소탐지 진단 → 6종 보정 시도
→ Q1 프로토타입 rescue 유효 확인 (GL 0-4s: 12→19, 인간 18).
mid-band rescue 오탐 판정. HANDOFF 생성, GitHub push 완료.

### 세션 2 (2026-08-08) — 청취 검증 + 천체 기법 응용 + local norm 돌파

- morphological opening: 실패 (하드코어 전 사건 충격적, filtered=0)
- SIR(u3): 보조 유효 (cry 억제, Q1 병용 필요)
- asinh 압축: 실패 (Otsu 동반 상승)
- **local norm(16s): VN 밀집 구간 돌파** (0→12-23)
- 청취 판정: VN=norm 압도적, SS=Otsu 최우수, GL=Q1, cry=추가검증

### 세션 3 (2026-08-08) — 2D onset map 실험 + block-gate 확립

- 2D onset map (mel spectrogram 영상처리): Fable 기각 — 음악 사건≠blob
- **block-gated adaptive detection**: Fable 제안, SExtractor 2-pass의 1D 재현. 채택.
  SS gate=Otsu 유지, VN 밀집구간 부분 구조 (20→183).
- gate+norm잔여 ≈ SS→Otsu, VN→norm 자동 적응 확인.
- VN 경계 블록 이슈 발견 (3:12 블록, 0.004 차이).

### 세션 4 (2026-08-08) — Dir 피아노 onset 탐지

#### 4-a. SuperFlux의 구조적 한계 확인

Dir 피아노곡에서 SuperFlux baseline = 등간격 클릭(피아노 타건 아님).
원인: max_filter(size=3)가 인접 배음으로 할인 + lag=2로 잔향 대비 차이 작음.
ONSET_BANDS(low=30-120, mid=120-2000, high=2000-20000)도 EDM 전제.

#### 4-b. onset function 탐색

| 방식 | 피크 | 판정 |
|---|---|---|
| SuperFlux baseline | 618 | 등간격 비트 artifact |
| 2-pass star suppression | +12 신규 | 유효하나 소규모 |
| log1p | 467~696 | 피아노 반응 있으나 미흡 |
| plain flux | 684 | 등간격 경향 |
| no-maxfilter | 690 | 선율 추적 |
| spectral novelty | 1052 | 멜로디 반응, burst 과다 |
| **novelty + norm (2s)** | **613** | **★ 현재 최선** |
| phase deviation | 932 | 음악 무관, 폐기 |
| per-band 독립 Otsu | 다수 | 오탐, 폐기 |

#### 4-c. 등간격 비트 진단

**B-1 교차대역 동시성**: "저역 전용 비트" 가설 기각.
618 피크 중 low_only = 11개(1.8%). 대부분 high만 활성.
원인: EDM 대역 정의에서 피아노 에너지가 high(2000Hz+)에 집중.

**주기성 진단**: 등간격의 정체 확인.
- IOI 중앙값 = 128ms. 120-130ms 빈에 305/590개(52%) 집중.
- 자기상관 지배 주기 = 255ms (235 BPM, ≈118 BPM 8분음표).
- 격자 피크 vs 비격자 피크: **스펙트럼 프로필 동일** (전 대역 <1dB 차이).
- **결론: 저역/배경음 유래가 아닌 곡의 리듬 격자에 대한 SuperFlux의 균일 반응.**

**v2 본선 영향 판정: 없음.**
EDM/하드코어에서 리듬 격자 반응 = 올바른 동작(킥/스네어가 격자 위에 있음).
피아노처럼 충격량이 약한 악곡에서만 "타건이 아닌 리듬 미세구조를 찍는" 문제 발현.
→ s4 고유 문제로 격리. 본선 파이프라인 수정 불필요.

#### 4-d. Fable 자문

[`Docs/s4_piano_advisory.md`](Docs/s4_piano_advisory.md) — 전문 보존.

핵심 제안:
- **B-2 CQT 기반**: mel→CQT 전환, 피아노 배음 분리 근본 개선
- **A-1 다중 스케일 WTMM**: 디아딕 스케일 사다리 정합
- **A-3 연속값 기하평균**: 이진 교집합 대신 `sqrt(f*n)` 후 Otsu
- **A-2 슬라이딩 percentile_filter**: 블록 경계 아티팩트 제거
- **B-3 Complex-domain flux**: 크기 가중 복소 거리

성운 프로젝트와의 대응: "공간축 × 색축" → "시간축 × 주파수축" 이중 판별.

#### 4-e. MIDI/악보 조사

| 소스 | 형식 | 상태 |
|---|---|---|
| MuseScore (Eriya Arai 편곡) | 피아노 솔로 악보 | 접근 가능 (팬) |
| nicoapple 귀카피 MIDI | MIDI | 링크 만료, 접속 불가 |

공식 MIDI/악보 없음. MuseScore 악보를 참조 정답으로 활용 가능 (D-21 비저촉).

---

## 3. 탐지 기법 인벤토리

### 확정 유효 (본선)

| 기법 | 원리 | 강점 | 약점 | D-21 |
|------|------|------|------|------|
| **Otsu 1차** | 전역 Otsu 이진 임계 | SS 최우수, 안정적 | 약한 사건 과소탐지 | 0 파라미터 |
| **Q1 rescue** | 윈도우 내 대역 프로토타입 유사도 Q1 | GL 피아노 12→19 | cry 과다, base<2 시 무력 | 0 파라미터 |
| **local norm(16s)** | 블록별 99-pct 자기 정규화 후 Otsu | VN 밀집 구간 압도적 | SS에서 기존보다 열세 | 블록크기=16s |
| **block-gate** | block_99pct < global_otsu → norm, else → otsu | SS 유지 + VN 구조 | VN 경계 블록 | 0 파라미터 |

### 확정 유효 (s4 피아노)

| 기법 | 원리 | 강점 | 약점 |
|------|------|------|------|
| **novelty + norm (2s)** | 코사인 거리 → bandpass+99pct norm → Otsu | baseline과 완전히 다른 피아노 사건 탐지 | 모든 타건 포착엔 미흡 |

### 보조 유효

| 기법 | 원리 | 역할 |
|------|------|------|
| **SIR(u3)** | 분광 균일도 ch_min/ch_max + 2-of-3 동시극대 | cry 과다탐지 억제, Q1 병용 |

### 기각

| 기법 | 사유 |
|------|------|
| mid-band rescue | 사용자 오탐 판정 |
| morphological opening (1D) | 하드코어 전 사건 충격적, filtered=0 |
| asinh 압축 | Otsu가 함께 상승, 역효과 |
| local Otsu | VN 폭발 |
| 2D onset map | Fable 기각: 음악 사건≠blob, CC 과다계수 |
| phase deviation (단독) | 음악 사건과 무관, 저에너지 잡음 위상에 휘둘림 |
| per-band 독립 Otsu | 오탐 |
| B-1 교차대역 동시성 (현재 대역) | EDM 대역 정의로 피아노 분리 불가 |

---

## 4. 파이프라인 (확정 후보)

```
오디오 → LUFS 정규화 → STFT → mel → SuperFlux 온셋 포락선
                                          ↓
                              block-gated detection
                              (visible → Otsu, invisible → local norm)
                                          ↓
                              Q1 prototype rescue
                                          ↓
                              [SIR(u3) conjunctive gate — 선택]
                                          ↓
                              4초 윈도우 계수 → 블라인드 설문
```

### block-gate 원리

SExtractor 2-pass 전략의 1D 재현 (Fable 제안):
1. 전역 Otsu 임계 계산
2. 각 16s 블록의 양수값 99-pct 계산
3. block_99pct < global_otsu → "전역에 보이지 않는" 블록 → local norm 탐지
4. block_99pct ≥ global_otsu → "전역에 보이는" 블록 → base Otsu 탐지
5. 전체 합산 후 진폭 내림차순 탐욕 선택 + 30ms 최소간격

---

## 5. 코드 상태

| 파일 | 상태 | 비고 |
|------|------|------|
| `src/config.py` | 안정 | |
| `src/audio_io.py` | 안정 | |
| `src/onset.py` | 안정 | SuperFlux + 3밴드 |
| `src/peak_pick.py` | **교체 필요** | peaks_with_mid → block-gate+Q1 |
| `src/pipeline.py` | **교체 필요** | |
| `src/sonify.py` | 안정 | |
| 기타 (counter, survey_gen 등) | 안정 | |

### 실험 스크립트

```
src/exp/s1_proto/   11개 — 세션 1 (Q1 프로토타입 탐색)
src/exp/s2_1d/      12개 — 세션 2 (SIR, local norm, morpho, asinh)
src/exp/s3_2d/       5개 — 세션 3 (2D onset map, block-gate)
src/exp/s4_piano/    8개 — 세션 4 (Dir 피아노 onset)
  _onset_piano.py       — 2-pass star suppression v2
  _onset_alt.py         — 4종 대안 onset (plain/nomaxf/novelty/perband)
  _onset_combine.py     — 3종 조합 (intersection/novelty_norm/plain_novelty)
  _onset_deburst.py     — novelty burst 억제 (wide gap/prominence)
  _onset_refine.py      — log1p/novelty × 블록크기 × deburst 전 조합
  _diag_crossband.py    — B-1 교차대역 동시성 진단 (가설 기각)
  _diag_periodicity.py  — 등간격 비트 주기성 진단 (리듬 격자 확인)
```

---

## 6. 규율

| ID | 규율 |
|---|---|
| D-07 | 에너지 계열 금지 — SuperFlux(스펙트럼 변화) 사용 |
| D-18 | 천장(r~0.85)과 바닥(r~0) 사전 정의 |
| D-21 | 출력 보고 파라미터 고르지 않는다 |
| D-24 | 관측과 해석을 분리 |
| D-v2-01 | 설문 완료 전 ground_truth.json 열지 않음 |
| D-v2-03 | 윈도우 4초 고정 |

---

## 7. 환경

```
Python 3.14.2  (C:\Python314)
.venv          v1과 공유 (E:\game\Music Hermeneutic AI\.venv)
GPU            RTX 3080 10 GB
```

콘솔 cp932 — 한글 출력은 `reconfigure(encoding="utf-8")`.
오디오: v2/audio/ 우선, v1 `audio/target/` 폴백.
대상 4곡: cry of viyella, Grievous Lady, Viyella's Nightmare, Swift Swing.

---

## 8. 기법별 수치 (참고용)

| 구간 | Otsu | Q1 | SIR(u3) | norm | gate | gate+Q1 | 인간 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GL 0-4 | 12 | **19** | 15 | 12 | 12 | ? | **18** |
| GL 16-20 | 6 | 9 | 10 | **10** | 6 | ? | **14** |
| VN 200-204 | 2 | **6** | 4 | 11 | ? | ? | **6** |
| VN 268-272 | 12 | 12 | 12 | **13** | 12 | ? | **14** |
| VN 3:00-3:04 | 0 | 0 | 0 | **19** | 19 | 27 | ? |
| cry 0-4 | 3 | 18(과다) | **14** | 3 | 3 | ? | ? |

### 전체 트랙 피크 수

| 트랙 | Otsu | Q1 | SIR | norm | gate | gate+Q1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| cry | 564 | 766 | 610 | 595 | 582 | 797 |
| GL | 591 | 638 | 629 | 582 | 591 | 638 |
| VN | 264 | 454 | 401 | **908** | 539 | 808 |
| SS | **558** | 693 | 587 | 502 | **558** | 693 |
