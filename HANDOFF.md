# HANDOFF — 2026-08-08 · block-gated adaptive detection 확립

이 문서는 **세션 경계 상태 + 연구 일지**다. 프로젝트 개요는 `README.md`.
v1 프로젝트(`E:\game\Music Hermeneutic AI\`)와는 독립된 후속 프로젝트.

**작업 이력의 전문은 아카이브에 있다** — `Docs/Archive/handoff/`.

---

## 0. 한 문단으로 — 지금 어디인가

**2D onset map 실험을 거쳐 block-gated adaptive detection으로 수렴했다.**
Fable 검토 결과 2D(mel spectrogram 영상처리)는 구조적 한계로 기각 —
음악 사건은 천체의 blob이 아니라 수직선/고조파 빗이므로 connected components가
과다계수. 대안으로 SExtractor 2-pass 전략의 1D 재현을 제안받아 구현·검증:
block 99-pct < global Otsu인 블록만 local norm으로 교체.
SS=순수 Otsu 유지, VN 밀집구간 부분 구조, GL=Q1 계층 필요.
소니파이 생성 완료, 사용자 청취 판단 대기.

---

## 1. 세션 이력

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

#### 3-a. 디렉터리 재편 + 스냅샷

스냅샷 커밋 `df4a664`. 실험 스크립트를 하위 디렉터리로 분리:
```
src/               ← 파이프라인 코어 (10개)
src/exp/s1_proto/   ← 세션 1 진단/소니파이 (Q1 프로토타입 탐색)
src/exp/s2_1d/      ← 세션 2 진단/소니파이 (SIR, local norm, morpho, asinh)
src/exp/s3_2d/      ← 세션 3 2D onset map 실험 + block-gate
```

#### 3-b. 2D onset map — 구조적 한계로 기각

GPT 대화에서 착안: mel spectrogram은 천체 사진과 동일한 2D 구조
(x=시간, y=주파수, 값=에너지). per-bin SuperFlux(합산 전)를 2D 행렬로 유지,
천체 기법을 2D에서 적용.

**테스트 1: Connected components**
- per-bin 99-pct 정규화 → 2D Otsu → morphological opening → CC labeling
- 결과: 648-6193개 과다탐지 (1D의 2-20배)
- 원인: 하나의 onset이 주파수 축에서 다수의 분리된 blob으로 깨짐

**테스트 2: 2D → 1D 투영**
- per-bin norm만 → frac/sum 투영 → 1D peak pick
- frac: 피아노(협대역)에 불리 (GL 0-4: 12→7)
- sum: VN 밀집 구간 여전히 0 (시간 축 dynamic range 미해결)

**테스트 3: 양축 정규화 (per-bin + per-block)**
- 6종 조합 시험: bin→blk, blk→bin, bin→sum→norm, bin→frac→norm, blk→sum
- 결과: **시간 축 블록 정규화가 핵심 동력. per-bin 추가 기여 미미.**
- `2D blk→bin→frac` ≈ `1D norm(16s)` (VN 밀집: 290 vs 298)

**Fable 검토 결론** (3가지):
1. 구현은 올바름. 천체 비유가 구조적으로 깨짐:
   - 천체: 광원=2D blob(PSF 집중) → CC로 계수 가능
   - 음악: 사건=수직선(퍼커시브) or 고조파 빗(피치) → blob이 아님
   - 주파수 축은 "몇 개"가 아니라 "어떤 종류" → 계수는 반드시 1D
2. per-bin 정규화: 조용한 빈의 노이즈 증폭 → 오히려 해로움
3. **대안 제안: block-gated adaptive detection** (SExtractor 2-pass의 1D 재현)

#### 3-c. block-gated adaptive detection — 원리적 성공

Fable 제안 구현·검증:

```
per 16s block:
  if block_99pct < global_otsu:   # 전역에 "보이지 않는" 블록
      → local norm 탐지 사용
  else:
      → base Otsu 탐지 사용
```

**D-21 준수**: block_99pct와 global_otsu는 이미 승인된 수치. 새 파라미터 0개.

**결과:**

| 트랙 | invisible 블록 | gate 결과 | 판정 |
|------|:---:|:---:|---|
| **SS** | 0/8 (0%) | gate=Otsu(558) | ✓ SS 최우수 유지 |
| **GL** | 0/9 (0%) | gate=Otsu(591) | Q1 계층 필요 (gate 자체는 변화 없음) |
| **cry** | 1/9 (11%) | gate=582 | 미세 차이 |
| **VN** | 5/19 (26%) | gate=539 | 밀집 구간 부분 구조 |

**VN 밀집 구간 (176-240s) 4초 윈도우별:**
```
Otsu:     [ 0,  0,  0,  0,  0,  0,  2,  5,  1,  4,  2,  0,  2,  2,  2,  0] = 20
norm:     [12, 19, 23, 17, 11,  5, 11, 21, 16, 24, 20, 21, 22, 22, 17, 37] = 298
gate:     [12, 19, 23, 17,  0,  0,  2,  5,  1,  4,  2,  0, 22, 22, 17, 37] = 183
gate+Q1:  [15, 28, 27, 18,  0,  0,  6,  7,  1,  9,  7,  0, 25, 22, 17, 37] = 219
```

**경계 이슈**: [3:12] 블록 99pct=2.827 ≈ Otsu 2.823 → 0.004 차이로 visible 판정
→ 4개 윈도우(3:08-3:24)가 0-5로 과소탐지. 설계 결함이 아닌 경계 문제.

#### 3-d. Fable 제안 — 전체 파이프라인 구조

```
base Otsu (전역)
  → block-gated local norm (invisible 블록만)
    → Q1 rescue (협대역 구조)
      → SIR(u3) conjunctive gate (Q1 프로토타입 불안정 시)
```

**stretch goal**: matched filtering — base Otsu 피크의 평균 포락선 형태를
template으로, 상관 후 임계 → cry 링잉 이중탐지 해결 가능성.

---

## 2. **다음 세션은 여기부터 읽는다**

### 현재 상태

- **block-gate 구현 완료**: `src/exp/s3_2d/_diag_blockgate.py`, `_sonify_blockgate.py`
- **소니파이 생성 완료**: `out/sonify/{트랙}/전체_gate_클릭.wav` 등 3종
- **pipeline 미통합**: 모든 기법이 진단 스크립트에만 존재
- **사용자 청취 판단 대기**

### 소니파이 파일 현황

```
out/sonify/{트랙}/
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
```

### 다음 단계

1. **사용자 gate 소니파이 청취 판정**
2. **경계 이슈 대응 결정** (VN 3:12 블록 0.004 차이)
3. pipeline 통합 → 설문 재생성 → 48개 응답 수집

### 열려 있는 문제

- VN [3:12] 블록 경계 이슈 (99pct ≈ Otsu, 4개 윈도우 과소탐지)
- cry Q1 과다탐지 (SIR 억제 필요)
- cry 링잉 이중 탐지 (matched filtering 후보)
- GL 16-20s 신스 킥 4회 미캐치
- peaks_with_mid 교체 필요 (pipeline.py)
- survey_gen.py append 버그
- 분류 모듈(classify.py) — 총 계수 검증 통과 후

---

## 3. 탐지 기법 인벤토리

### 확정 유효

| 기법 | 원리 | 강점 | 약점 | D-21 |
|------|------|------|------|------|
| **Otsu 1차** | 전역 Otsu 이진 임계 | SS 최우수, 안정적 | 약한 사건 과소탐지 | 0 파라미터 |
| **Q1 rescue** | 윈도우 내 대역 프로토타입 유사도 Q1 | GL 피아노 12→19 | cry 과다, base<2 시 무력 | 0 파라미터 |
| **local norm(16s)** | 블록별 99-pct 자기 정규화 후 Otsu | VN 밀집 구간 압도적 | SS에서 기존보다 열세 | 블록크기=16s |
| **block-gate** | block_99pct < global_otsu → norm, else → otsu | SS 유지 + VN 구조 | VN 경계 블록 | 0 파라미터 |

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
| **2D onset map** | Fable 기각: 음악 사건≠blob, CC 과다계수, per-bin norm 역효과 |

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
  _diag_onset2d.py          — 2D CC (기각)
  _diag_onset2d_proj.py     — 2D→1D 투영 (기각)
  _diag_onset2d_dual.py     — 양축 정규화 (1D norm과 수렴)
  _diag_blockgate.py        — block-gate 진단 (채택)
  _sonify_blockgate.py      — block-gate 소니파이 (채택)
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

**SS gate=Otsu**: norm 퇴보 완전 방지.
**VN gate**: 밀집 구간 부분 구조 (20→183, norm 298).
