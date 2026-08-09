# HANDOFF — 2026-08-10 · sculpt 청취 판정: HPSS percussive 주 후보

이 문서는 **세션 경계 상태 + 연구 일지**다. 프로젝트 개요는 `README.md`.
v1 프로젝트(`E:\game\Music Hermeneutic AI\`)와는 독립된 후속 프로젝트.

**작업 이력의 전문은 아카이브에 있다** — `Docs/Archive/handoff/`.
직전 스냅샷(세션 11까지, A-2+posdist 395 + BS 전사 청취 대기):
[`Docs/Archive/handoff/HANDOFF_2026-08-09_session11_a2-posdist395-bs-transcription.md`](Docs/Archive/handoff/HANDOFF_2026-08-09_session11_a2-posdist395-bs-transcription.md).

---

## 0. 한 문단으로 — 지금 어디인가

**본선 4곡은 block-gated adaptive로 수렴, pipeline 통합 대기(변경 없음).**
s4 sculpt 3패스 청취 판정: **`hpss_percussive` = 주 후보**(타건 순간 포괄).
`lpc_residual`은 이산성은 좋으나 볼륨 편차·누락 의심 → 값 조정 테스트 시
**보조 후보**. `lpc_synthesis`는 사건/링잉 대비가 raw보다 돋보여 유효.
harmonic·sine_tonal은 전자피아노/배음 포락선 인상. sine_residual은 울림·링잉
커 관심에서 다소 멀지만 중저역 강조 활용은 미결.

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
> Fable 자문 전문 + 두 방향 정리 + 성운 프로젝트 대응표. 피아노 외에도 참고 가치.

**목표**: `102 - Dir.wav` (dai, 오르골/피아노, ~117s) — 피아노 건반 타건 전수 탐지.

**비교 기준선 (ODF 경로, 동결)**: A-2 + positive rescue **395**. 작업 산출에
395 대비 소니파이는 넣지 않음(사용자 청취 비교).

**stem event sculpt** (`out/stems/Dir/event_sculpt/`):

| 산출 | 청취 판정 |
|------|-----------|
| **hpss_percussive** | **주 후보** — 건반 타건 순간을 담았다고 봄 |
| **lpc_residual** | **보조 후보(조건부)** — 이산 사건 감지 성공, 볼륨 편차·누락 의심 → 값 조정 테스트 포함 시 |
| **lpc_synthesis** | **유효** — sine_tonal보다 사건만 남기려는 인상; 링잉 있으나 사건↔링잉 볼륨 대비가 raw보다 돋봄 |
| hpss_harmonic | 전자피아노 인상; sine_residual보다 raw에 가까워 **가벼운 전처리 단계**로 고려 가능 |
| sine_residual | 피아노 울림·**링잉 큼** → 이번 관심에서 다소 멀음; 중저역 강조 활용은 미결 |
| sine_tonal | 전자피아노·포락선처럼 이어짐 → 사건보다 **배음**으로 들림 (타 용도 여지, 본 프로젝트는 의문) |

**s4 다음 단계**:

1. **주 후보 `hpss_percussive` 기준** 다음 점진 패스/계수·표현 설계.
2. (선택) `lpc_residual` 고정값 조정 스윕은 D-21 절차를 새로 선언한 뒤에만.
3. `lpc_synthesis` 유효성 활용 여부(대비 강화 단 vs 단독) 결정.

**B-2 CQT 판정: 기각.** 800 peaks 중 짧은 burst가 크게 증가했고 사용자 청취에서
CQT 전용 클릭이 주로 burst·링잉·오탐으로 판정됨. 세션 5 참조.

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
  전체_cqt_n2s_클릭.wav         800  B-2 CQT novelty (기각)
  전체_cqt_n2s_vs_mel_n2s_비교_클릭.wav
                                    3kHz=공통, 5kHz=CQT전용, 1.5kHz=mel전용
  cqt_b2_metrics.json                B-2 고정 조건 및 진단 수치
  전체_wtmm_nov_클릭.wav        691  A-1 (청취 불충분)
  전체_fusion_n2s_클릭.wav       351  ★ A-3 기하평균 융합 (채택)
  전체_wtmm_fusion_클릭.wav      419  A-1+A-3 (청취 불충분)
  전체_*_vs_nov_n2s_비교_클릭.wav
                                    3kHz=공통, 5kHz=변형전용, 1.5kHz=기존전용
  wtmm_fusion_metrics.json           A-1/A-3 고정 조건 및 진단 수치
  전체_fusion_slide_n2s_클릭.wav 355  A-2 sliding (판정 보류)
  전체_fusion_slide_n2s_vs_fusion_n2s_비교_클릭.wav
  sliding_norm_metrics.json           A-2 경계/주기성 진단
  전체_complex_slide_클릭.wav    450  B-3 단독
  전체_nov_complex_slide_클릭.wav 450 novelty×complex
  전체_tri_complex_slide_클릭.wav 306 novelty×flux×complex (청취 최선)
  전체_*_vs_fusion_slide_비교_클릭.wav
  전체_fusion_slide_complex_attribution_클릭.wav
                                    3kHz=phase 지원, 5kHz=phase 미지원
  complex_hysteresis_metrics.json     B-3/meso-burst 귀속 진단
  전체_core_a2_complex_only_클릭.wav 276  정밀도 보조
  전체_residual_a2_only_클릭.wav      79  역할 불명/신뢰 곤란
  전체_rescue_complex_only_클릭.wav  174  선별 필요
  전체_carved_클릭.wav                334  source carving (기각)
  전체_carved_removed_only_클릭.wav    21
  전체_a2_vs_complex_stereo.wav            L=A-2, R=complex
  전체_carved_vs_a2_비교_클릭.wav
  source_carving_metrics.json
  전체_posdist_flux_slide_클릭.wav        206  positive distribution×flux
  전체_posdist_flux_slide_vs_fusion_slide_비교_클릭.wav
                                         공통=166, positive전용=40, A-2전용=189
  전체_posdist_flux_positive전용_클릭.wav 40   실제 타건 rescue 후보
  전체_posdist_flux_A2전용_클릭.wav       189  대부분 실제 타건
  전체_a2_posdist_rescue_클릭.wav         395  A-2 보존+positive rescue
  posdist_metrics.json
  전체_stem_*_piano_candidate395_클릭.wav    3모델 piano stem 위 395
  전체_stem_support395_비교_클릭.wav         3/3=117, 2/3=76, 0-1/3=202
  전체_stem_consensus_missed_클릭.wav        42
  전체_bs_reference_vs_candidate395_비교_클릭.wav
                                             공통=190, BS-only=65, 395-only=203
  stem_consensus_metrics.json

out/stems/Dir/
  {bs_roformer,spleeter,demucs}/             전체 stems+piano+residual
  stem_manifest.json                         모델/버전/checkpoint SHA-256
  stem_validation_metrics.json               재구성·상관·형식 검증
  event_sculpt/                              ★ sculpt 3패스 청취 WAV
    hpss_percussive.wav   ★ 주 후보 (타건 순간)
    hpss_harmonic.wav        전자피아노; 가벼운 전처리 후보
    lpc_residual.wav         보조 후보(값 조정 테스트 시)
    lpc_synthesis.wav        유효 (사건/링잉 대비)
    sine_residual.wav        울림·링잉 큼; 중저역 미결
    sine_tonal.wav           배음 포락선; 본 프로젝트 의문
    sculpt_manifest.json / sculpt_determinism.json
  전체_log1p_raw_클릭.wav       696  log1p (raw)
  ... 외 다수 (블록크기×deburst 조합, intersection, 비교)
```

---

## 2. 세션 이력

### 세션 13 (2026-08-10) — stem event sculpt 3패스 병행 산출 + 청취 판정

**실행**: `run_passes.py` + `--determinism-check`. 입력 BS piano 전장 116.813s.
WAV SHA 재실행 일치.

**고정 파라미터**:
- HPSS: kernel=31, power=2, margin=1, n_fft=2048, hop=256
- LPC: order=24, frame=2048, hop=512, pre-emphasis=0.97
- Sinusoidal: freq local-max ∧ mag≥frame p90, n_fft=2048, hop=256

**관측(수치)**:
- source peak=1.078 rms=0.184
- hpss_percussive peak=0.536 rms=0.037; harmonic peak=0.960 rms=0.162
- lpc_residual peak=0.534 rms=0.009; synthesis peak=0.634 rms=0.023
- sine_residual peak=0.638 rms=0.087; tonal peak=0.638 rms=0.105

**사용자 청취 판정**:
- `hpss_percussive`: **주 후보** — 생각했던 피아노 건반 치는 순간을 담음.
- `lpc_residual`: 더 이산적으로 피아노 사건 감지에 성공. 다만 볼륨 편차 크고
  누락 의심 → **값 조정 테스트를 포함할 때 보조 후보**.
- `hpss_harmonic`: 전자피아노로 친 듯한 소리. `sine_residual`보다 raw에 가까워
  **가벼운 전처리 단계**로 고려할 만함.
- `lpc_synthesis`: `sine_tonal`보다 사건만 남기려 한 인상. 링잉 포함이나
  사건↔링잉 볼륨 대비가 raw보다 돋보여 **유효**.
- `sine_residual`: 피아노 울림 포함 → 이번 관심사에서 다소 멀 수 있음.
  중저역 강조 활용 여지는 미결. **링잉이 큼**.
- `sine_tonal`: 전자피아노 인상. 사건성보다 포락선처럼 이어져 청자에게
  **배음**으로 읽힘 (배음 쪽 활용 여지 있으나 본 프로젝트는 의문).

**해석**: 1차 병행 산출의 주선은 HPSS percussive. LPC residual은 이산성
이득이 있으나 고정값 그대로는 보조에 그침. sinusoidal 잔여/톤은 사건 축보다
울림·배음 축에 가깝다. 이번 청취로 파라미터를 바꾸지 않음(D-21).

### 세션 12 (2026-08-10) — stem event sculpt 작업영역 준비

- 직전 HANDOFF를
  `Docs/Archive/handoff/HANDOFF_2026-08-09_session11_a2-posdist395-bs-transcription.md`
  로 아카이브.
- 새 작업영역 `src/exp/s4_piano/stem_event_sculpt/` 생성 (README + .gitignore).
- ODF 395 경로는 비교 기준선으로 동결. 전처리·소니파이 계획/구현은 미착수.

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

### 세션 5 (2026-08-08) — B-2 CQT novelty 기각

**고정 조건**: A0(27.5Hz), 12 bins/octave, 108 bins(9옥타브), tuning=0.
mel/CQT 모두 cosine novelty → 2s smooth 제거 → 2s block 99-pct norm
→ Otsu → 30ms 최소간격을 동일 적용. 출력 기반 튜닝 없음.

**관측**:
- mel 대조군 613 peaks 재현, CQT 800 peaks.
- ±30ms 일대일 매칭: 공통 448, CQT 전용 352, mel 전용 165.
- 120-130ms IOI 집중률: mel 28.0% → CQT 15.2%.
- 100ms 미만 IOI: mel 88 → CQT 309, 50ms 미만: 35 → 106.
- 지배 자기상관 주기: mel 255ms → CQT 1015ms.
- 전체 트랙 재실행에서 수치·WAV·JSON SHA-256 동일.

**해석**: CQT는 기존 128ms 리듬 격자 집중을 줄였으나, 더 심한 단시간
다중 피크를 만들었다. 배음 분리는 개선될 수 있어도 인접 프레임 cosine
novelty가 CQT 빈의 세부 변동과 링잉에 과민 반응한 것으로 보인다.

**사용자 청취 판정: 기각.** CQT 전용(5kHz)은 주로 burst·링잉·오탐.
총 피크 수 증가는 타건 복구가 아니며, D-21에 따라 CQT 파라미터 재튜닝 없이
B-2를 종료한다. 다음 우선순위는 A-1/A-3.

### 세션 6 (2026-08-08) — A-3 기하평균 융합 채택

**고정 조건**:
- A-1: 0.25/0.5/1/2/4/8s 사다리, 인접 스케일 ±30ms 일대일 chain,
  2스케일 이상 지속, 마지막 30ms 선택.
- A-3: 2s에서 `sqrt(novelty_norm * no-maxfilter_flux_norm)`, Otsu 1회.
- 결합형: 각 스케일에서 기하평균 후 A-1 chain.

**관측**:
- 대조 `nov_n2s`: 613 peaks, <100ms IOI 88, AC 255ms.
- A-1 `wtmm_nov`: 691 peaks. 대조 공통 604, 전용 87, 대조 전용 9.
  <100ms IOI 111, AC 255ms. 691개 중 507개가 6스케일 모두 지속.
- A-3 `fusion_n2s`: 351 peaks. 공통 293, 전용 58, 대조 전용 320.
  <50/<100ms IOI 모두 0, AC 1015ms.
- 결합 `wtmm_fusion`: 419 peaks. 공통 326, 전용 93, 대조 전용 287.
  <50ms 0, <100ms 1, AC 1015ms.
- 전체 트랙 재실행에서 수치·WAV·JSON SHA-256 동일.

**해석**:
- novelty 극대 대부분이 전 스케일에서 지속되어 A-1 chain만으로는 artifact와
  타건을 충분히 분리하지 못했고, 오히려 후보가 증가했다.
- A-3은 두 연속 검출자가 동의하는 사건만 보존해 짧은 burst와 255ms 격자
  반응을 강하게 억제했다. 다만 0-4s가 0개이고 대조 피크 320개가 소실되어
  정량상 과소탐지 위험은 남는다.
- 결합형은 A-3보다 68개 많고 일부 개선도 들리지만, 청취상 burst가 함께
  나타나 이득이 혼재했다. 수치상 <100ms IOI 1개와 청감상 burst를 구분해 기록한다.

**사용자 청취 판정**:
- A-1: 불충분.
- A-3: **채택 — 세 변형 중 `전체_fusion_n2s_vs_nov_n2s_비교_클릭.wav`가 가장 나음.**
- A-1+A-3: 불충분 — 개선과 burst가 혼재.

출력을 보고 스케일·정합·융합식을 재조정하지 않는다. 다음은 채택된 A-3에
A-2 sliding percentile을 적용해 고정 블록 경계를 제거한다.

### 세션 7 (2026-08-08) — A-2/B-3 유망, 최종 용도 보류

#### 7-a. A-2 sliding percentile

**관측**:
- block A-3 351 재현, sliding 355. 공통 327, sliding 전용 28, block 전용 24.
- <100ms IOI 양쪽 0, AC 양쪽 1015ms.
- 2초 경계 jump p95: 0.15596 → 0.16019, 경계 ±30ms 피크: 10 → 10.

**해석/청취**:
- sliding은 경계 불연속을 수치상 개선하지 않았고 검출 집합도 거의 동일하다.
- 단독 `전체_fusion_slide_n2s_클릭.wav`는 소실이 많지만 더 확실한 피아노
  사건을 드러낸다는 의의가 있음. 다만 생략된 타건이 많아 최종 판정 보류.
- 비교 파일은 block/sliding 합집합을 음높이로 구분하므로 단독 파일과
  클릭 발생 차이가 거의 없는 것이 정상이다.

#### 7-b. 사건 히스테리시스 연역

- A-2는 건반 타격 전용이 아니라 `spectral shape change ∧ positive flux` 사건:
  페달, 배음 재분배, 링잉 재상승도 포함 가능.
- <100ms IOI는 0이지만 100-300ms 간격은 220개.
- Otsu 초과 run 365개 중 다중 피크 run은 0개 → plateau collapse/gap 반복은
  원인을 건드리지 않음. 청감 burst는 분리된 meso-scale excursion으로 재정의.

#### 7-c. B-3 complex-domain flux

**관측**:
- complex 단독 450: <100ms 84, 0-4s 11.
- novelty×complex 450: <100ms 53, 0-4s 14.
- novelty×no-max flux×complex 3-way 306: <100ms 0, 0-4s 1.
- 3-way vs A-2: 공통 289, 3-way 전용 17, A-2 전용 66.
- A-2 355개 중 phase-supported 276, unsupported 79.
- 100-300ms follower 220개 중 phase-supported 176, unsupported 44.
- 전체 트랙 재실행에서 수치·WAV·JSON SHA-256 동일.

**해석/청취**:
- meso-burst 대부분도 complex 위상 증거를 동반해 “위상 리셋 없는 novelty
  꼬리”라는 단순 가설은 약화. complex ODF 자체도 링잉 재상승에 반응할 수 있음.
- attribution 청취에서는 실제 burst가 줄어든 것으로 감상됨.
- 세 B-3 변형 중 **3-way `tri_complex_slide`가 최선**.
- 사용자 판정: 주 후보/보조 유효 모두 가능할 정도로 피아노 이벤트를 추종.
  단독과 기존 기법 결합 중 어느 쪽이 나은지는 후속 소니파이로 결정하며,
  현재 용도를 제한하지 않는다.

다음 설계는
`E:\game\2Test1\Docs\references\stellar\fits_work\veil_nebula\baseline\BASELINE.md`
의 성운 추출 방법에서 이중 판별과 보수적 억제 원리를 다시 참조한다.

### 세션 8 (2026-08-08) — source-derived event carving 기각

**성운 baseline 대응**:
- strict tri-complex를 최종 출력이 아닌 aggressive-derived soft mask로 사용.
- `tri_mask = clip(tri / tri_otsu, 0, 1)`.
- `carved = fusion_slide * sqrt(tri_mask)`.
- 새 극대점을 만들지 않고 기존 A-2 피크만 carved Otsu로 보수적 감쇠.

**관측**:
- A-2 355, complex 450, tri 306 재현.
- A2∩complex 276, A2-only 79, complex-only 174.
- carved 334, 감쇠 21. A-2의 완전한 부분집합.
- A-2 meso follower 220개 중 carved 생존 207, 감쇠 13.
- 전체 트랙 재실행에서 수치·WAV·JSON SHA-256 동일.

**사용자 청취 판정**:
- A2∩complex 276: **정밀도 보조**. 완전히 확실하지는 않지만 피아노 사건을
  추종하며, 타건 누락이 많아 주 검출기로는 부족.
- A2-only 79: 신뢰하기 어려움. 감쇠 대상인지 실제 사건과 artifact의 혼재인지도
  명확히 분류하기 어려워 역할 보류.
- complex-only 174: 선별 필요. 유효 사건과 비피아노 burst가 혼재하며,
  사용자는 novelty가 놓친 원하는 사건을 충분히 포함하지 않는 것으로 감상.
- source carving 334: **기각 — 감쇠된 21개에 유효 타건이 많음.**

**해석**:
- tri 증거 부족을 곧바로 제거 근거로 쓸 수 없다. 성운의 source-derived
  darkness와 달리 complex phase evidence는 피아노 사건의 안전한 부정 증거가 아님.
- A2∩complex는 precision layer로는 유효하지만, complex-only 자동 rescue와
  tri-mask carving은 안전하지 않다.
- 병목은 후단 carving보다 novelty 정의가 원하는 사건을 충분히 제시하지 못하는
  데 있을 가능성이 커짐. 다음은 positive bin-wise novelty.

### 세션 9 (2026-08-08) — positive distribution novelty

**정의**:
- log-mel을 `[0,1]` activation으로 이동한 뒤 프레임별 L1 분포로 정규화.
- `sum(max(p[t] - p[t-1], 0))`로 증가한 spectral mass만 측정.
- 단순 positive log-mel flux 반복이 아니라 진폭 불변 distribution 변화량.

**관측**:
- positive 단독 571: 120–130ms 41.1%, AC 255ms로 리듬 격자 반응이 악화.
- positive×flux 206: 120–130ms 11.8%, 0–4s 3개.
- A-2 355와 비교: 공통 166, positive 전용 40, A-2 전용 189.
- positive×flux×complex 167: tri-complex 대비 공통 148, 전용 19, 누락 158.
- A-2 + positive 전용 rescue = 395: <100ms 5, 0–4s 3.
- 전체 트랙 재실행에서 WAV·JSON 결정성 확인.

**사용자 청취 판정/해석**:
- A-2 전용 낮은 클릭 189개는 대부분 실제 타건. positive×flux를 주 검출기로
  대체하면 유효 타건을 과도하게 잃으므로 **대체안은 기각**.
- positive 전용 40개도 실제 타건으로 판정. positive distribution은 억제기가
  아니라 cosine/A-2가 놓친 사건을 더하는 **보완 rescue로 유효**.
- 따라서 A-2 원형을 보존하고 positive 전용만 합친 395개를 다음 청취 후보로 둔다.

### 세션 10 (2026-08-09) — 3-model piano stem validation

**실행/재현성**:
- `src/exp/s4_piano/stem_validation/` 독립 작업공간에 Python 3.11, 모델,
  임시 추론물을 격리. 기존 공유 환경은 수정하지 않음.
- BS-Roformer SW 6-stem, Spleeter 5-stem, HTDemucs 6-source 전체 stem 생성.
- canonical 44.1kHz stereo FLOAT WAV와 모델 ID/버전/checkpoint SHA-256 저장.
- 재실행에서 canonical stem SHA-256, 소니파이 SHA-256 및 사건 수 동일.

**관측**:
- stem A-2+positive rescue: BS 255, Spleeter 279, Demucs 236.
- 395 후보의 모델 지지: 3/3 117, 2/3 76, 1/3 86, 0/3 116.
- 2개 이상 stem 모델 합의 사건 234, 395가 포함 192, 누락 42.
- 기존 후보의 합의 사건 포괄률: nov 73.5%, A-3 75.6%, A-2 77.8%,
  tri-complex 71.4%, **A-2+positive rescue 82.1%로 최상**.
- BS↔Demucs piano waveform r=0.958, RMS-envelope r=0.980.
- BS 주 참조 255 vs 395: 공통 190, BS-only 65, 395-only 203.

**사용자 청취 판정**:
- BS residual은 복합 드럼·기타 사건 위주이며 피아노 혼입은 있으나 주되지 않음.
- BS piano stem은 피아노 타건이 집중되고 residual에 피아노가 많이 남지 않으며,
  오인 분리는 주되지 않고 기타 링잉 혼입이 일부 있음.
- 세 모델 중 **BS-Roformer가 가장 우수**. Spleeter/Demucs는 품질이 떨어지나
  서로 다른 사건을 잡으므로 보조 감사 레이어로 가치가 있음.

**해석/결정**:
- BS piano stem을 이 곡의 주 귀속 참조로 사용한다. 단, 분리 모델과 stem 위에
  같은 onset 정의를 적용한 결과이므로 절대 ground truth나 튜닝 목표로 쓰지 않는다.
- 다음은 BS-only/395-only 청취와 BS stem의 독립 piano transcription/MIDI를 통해
  현재 395가 실제 최선인지 검증한다.

### 세션 11 (2026-08-09) — BS stem 독립 Piano-to-MIDI 검증 (청취 대기)

**실행/재현성**:
- 입력은 `out/stems/Dir/bs_roformer/piano.wav` 하나로 고정.
- Transkun 2.0.1 기본 V2 checkpoint를 주 전사기로, Basic Pitch 0.4.0
  기본 ICASSP 2022 모델을 감사기로 사용. threshold/segment 사후 조정 없음.
- 원시 note event와 MIDI를 보존하고, 비교용으로만 30ms complete-linkage
  note-on cluster 및 median 대표시각을 생성.
- 독립 재실행에서 두 모델의 note event, cluster, Basic Pitch raw event가 모두 동일.
  checkpoint/config/output SHA-256과 기본 설정을 manifest에 기록.

**관측(±30ms 일대일, 정답률 아님)**:
- Transkun: 1,760 notes / 765 clusters. Basic Pitch: 1,392 / 727.
- 두 전사 공통 592, Transkun-only 173, Basic-Pitch-only 135.
  Transkun 기준 합의율 77.4%, Basic Pitch 기준 81.4%; 시각 오차 중앙값 6.1ms.
- Transkun 대비 395: 공통 261, Transkun-only 504, 395-only 134.
  `reference coverage` 34.1%, `reference-supported fraction` 66.1%.
- Basic Pitch 대비 395: 공통 278, Basic-Pitch-only 449, 395-only 117.
  coverage 38.2%, supported fraction 70.4%.
- 두 전사가 공통으로 찾았으나 395가 놓친 사건은 346개.
- ±50ms 민감도에서 Transkun 대비 395 supported fraction이 90.6%로 상승하고,
  ±30ms 매칭의 절대 시각 오차 중앙값이 22.9ms이므로 기존 검출기와 전사기 사이
  체계적 시간 정렬 차이가 큰 변수다. D-21에 따라 보정하거나 허용치를 재선택하지 않음.

**현재 상태**:
- 8개 역할별 WAV를 원곡/BS piano stem 양쪽에 생성하고 형식·길이·clipping·
  SHA-256 검증 완료.
- 위 수치는 true precision/recall이 아니다. BS stem 공유 입력의 전사 참조에 대한
  coverage/support일 뿐이며 **395의 지위와 전사 참조 신뢰성은 사용자 청취 대기**.
- 청취 후에만 395 최종 지위와 MuseScore 구조 정렬 진입 여부를 결정한다.

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
| **A-3 fusion (2s)** | `sqrt(novelty_norm × no-max flux_norm)` → Otsu | burst/격자 반응 억제, 사용자 최선 판정 | 0-4s 미탐, 과소탐지 위험 |
| **HPSS percussive (sculpt)** | BS piano → anisotropic median P | 타건 순간 포괄, sculpt **주 후보** | 이후 점진 축소·계수 미착수 |

### 보조 유효

| 기법 | 원리 | 역할 |
|------|------|------|
| **SIR(u3)** | 분광 균일도 ch_min/ch_max + 2-of-3 동시극대 | cry 과다탐지 억제, Q1 병용 |
| **A-2 sliding A-3** | block p99 → sliding p99 | 확실한 피아노 사건 중심, 최종 용도 보류 |
| **B-3 tri-complex** | novelty×flux×complex 연속값 3-way | 피아노 이벤트 추종 유망, 주/보조 용도 보류 |
| **A2∩complex core** | ±30ms 일대일 공통 사건 | 타건 누락 큰 정밀도 보조 |
| **positive-distribution rescue** | A-2와 비매칭인 positive×flux 사건 추가 | A-2 보존 + 실제 타건 40개 복구 |
| **BS-Roformer piano stem** | 동일 원음 6-stem 분리 | 주 귀속 참조; Spleeter/Demucs로 감사 |
| **LPC residual (sculpt)** | order=24 화이트닝 여기 | 이산 사건 감지; 볼륨 편차·누락 → 값 조정 시 보조 |
| **LPC synthesis (sculpt)** | LPC 재합성 | 사건↔링잉 대비 raw보다 돋봄, 유효 |
| **HPSS harmonic (sculpt)** | H 성분 | 전자피아노 인상; raw에 가까운 가벼운 전처리 후보 |

### 불충분 (s4 피아노)

| 기법 | 판정 |
|------|------|
| A-1 WTMM-inspired chain | 대조 대부분 유지+87개 추가, artifact 분리 이득 불명확 |
| A-1+A-3 결합 | 일부 개선과 burst가 혼재해 단독 채택 불충분 |
| A2-only residual | 실제 사건/artifact 여부와 감쇠 안전성 불명 |
| complex-only rescue | 유효 사건과 비피아노 burst 혼재, 선별 필요 |
| sine_residual (sculpt) | 울림·링잉 큼, 관심에서 다소 멀음; 중저역 활용 미결 |
| sine_tonal (sculpt) | 배음 포락선으로 들림; 본 프로젝트 사건 축은 의문 |

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
| B-2 CQT novelty | 리듬 격자 집중은 감소했으나 짧은 burst 급증, 사용자 오탐 판정 |
| source-derived event carving | 감쇠 21개에 유효 타건 다수, complex 부족은 안전한 부정 증거가 아님 |
| positive distribution 주 검출기 | A-2 전용 실제 타건 189개 중 다수를 누락; rescue로만 유효 |

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
src/exp/s4_piano/   13개 — 세션 4-9 (Dir 피아노 onset)
  _onset_piano.py       — 2-pass star suppression v2
  _onset_alt.py         — 4종 대안 onset (plain/nomaxf/novelty/perband)
  _onset_combine.py     — 3종 조합 (intersection/novelty_norm/plain_novelty)
  _onset_deburst.py     — novelty burst 억제 (wide gap/prominence)
  _onset_refine.py      — log1p/novelty × 블록크기 × deburst 전 조합
  _onset_cqt.py         — B-2 CQT novelty + mel 대조/주기성 진단 (기각)
  _onset_wtmm_fusion.py — A-1/A-3 분해 비교 (A-3 채택)
  _onset_sliding_norm.py — A-2 sliding p99 (판정 보류)
  _onset_complex_hysteresis.py — B-3 complex + meso-burst 귀속
  _onset_source_carving.py — tri mask 기반 보수적 carving (기각)
  _onset_posdist.py    — positive distribution novelty + A-2 rescue
  _diag_crossband.py    — B-1 교차대역 동시성 진단 (가설 기각)
  _diag_periodicity.py  — 등간격 비트 주기성 진단 (리듬 격자 확인)
  stem_validation/      — 3-model piano stem 분리·합의·소니파이·검증
    transcription/      — Transkun/Basic Pitch 독립 전사·평가·소니파이·검증
  stem_event_sculpt/    — HPSS percussive 주 후보; LPC 보조/유효 (청취 판정)
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
