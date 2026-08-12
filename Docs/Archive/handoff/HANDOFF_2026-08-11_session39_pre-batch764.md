# HANDOFF — 2026-08-10 · Dir 통합본 764 채택 (506∪전체_adaptive)

이 문서는 **세션 경계 상태 + 연구 일지**다. 프로젝트 개요는 `README.md`.
v1 프로젝트(`E:\game\Music Hermeneutic AI\`)와는 독립된 후속 프로젝트.

**작업 이력의 전문은 아카이브에 있다** — `Docs/Archive/handoff/`.
직전 스냅샷(세션 38까지 · Dir 통합본 764):
[`Docs/Archive/handoff/HANDOFF_2026-08-10_session38_dir-union764-506-adaptive.md`](Docs/Archive/handoff/HANDOFF_2026-08-10_session38_dir-union764-506-adaptive.md).
그 이전(세션 11까지):
[`Docs/Archive/handoff/HANDOFF_2026-08-09_session11_a2-posdist395-bs-transcription.md`](Docs/Archive/handoff/HANDOFF_2026-08-09_session11_a2-posdist395-bs-transcription.md).

---

## 0. 한 문단으로 — 지금 어디인가

**본선 4곡은 block-gated adaptive로 수렴, pipeline 통합 대기(변경 없음).**
Dir **현 시점 통합본 = 764** (`506 ∪ Dir 전체_adaptive`, ±30ms).
역할: **506 = 피아노 사건**, **전체_adaptive = 저역 비트·기타 구조적 사건**
(단독 소니파이·비교 청취 근거). 피아노 전용 최선 클릭은 계속 **506**.
**선결**: 764의 stem합의 **missed 2** 청취 판정
(`stem_consensus_234_missed_by_764_*` / vs764 freqsep 산출됨).
527(2k×21)·union∪miss3(615) miss-3는 보류 유지.

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
> **파이프라인 소개 (adaptive · 506 · 764)**: [`Docs/dir_764_pipeline.md`](Docs/dir_764_pipeline.md)

**목표**: `102 - Dir.wav` — 피아노 건반 타건 전수 탐지.
ODF **395**는 비교 기준선(작업 산출에 395 A/B 소니파이 없음).

**pass2 / LPC 잔여 판정** (세션 14→21):
- soft_gate_removed 기각; perc 클릭 3종은 병행 후보로 유지
- LPC o12 residual: `rms_plain` 신뢰 가능; adaptive는 과거 최선이었으나
  **세션 34에서 506에 양보**

**Dir 통합본 (세션 38 채택) = 764**:
`506 ∪ out/sonify/Dir/전체_adaptive_클릭` (SuperFlux+peaks_adaptive, n=679).
공통 421 / 506-only 85 / adaptive-only 258. stem합의 cov **99.1%**(분모 234),
**missed 2** (common 232).

청취 판정(역할 분담):
- **506** — 피아노 건반 사건에 더 특화
- **전체_adaptive** — 저역 비트·다른 구조적 사건에 더 특화
- 단독 소니파이도 청취 후 위 역할을 확정; 통합본으로 **764** 채택

소니파이(대표):
- piano low: `…/cmp506_vs_dirAdaptive_low_g0p20_{unified3k|freqsep}_클릭_p764_*.wav`
- 원곡 bed: `…/cmp506_vs_dirAdaptive_origmix_{raw|lufs}_g{1p00|0p20}_{unified3k|freqsep}_*.wav`
  (`origmix_g*` = raw 별칭; `lufs` = `load_mono`/TARGET_LUFS = `전체_adaptive_클릭`과 동일 레벨 처리)

**피아노 전용 최선 클릭 (세션 34, 유지)**:
`…/fusion_kenv_agree_only_on_piano_클릭_p506.wav`
(+ low / freqsep `…_p506*`)

정의: `perc_tilt_k_env_adaptive`(502) ∪ LPC-order agreement 전용(±30ms 밖) **4**점.
**2k o12-deburst(21) 제외.**

**보류**: `fusion_kenv_agree_o12db_*_p527*` (2k×21 포함) — 유효/비유효 혼재.
miss-3(`0:21.142`, `0:21.525`, `1:32.816`)·615 — 비치명/의미 보류.

**신뢰 기준선**:
`…/lpc_o12_residual_rms_plain_클릭.wav` (181)

**이전 틸트 단독**:
`perc_tilt_k_env_adaptive` 502 / `perc_tilt_k_env` 340

**s4 다음 단계**:

1. **선결**: 764 stem합의 **missed 2** 청취 판정
   (`…/stem_consensus_234_missed_by_764_low_g0p20_클릭_p2.wav`,
   `…/stem_consensus_234_vs764_low_*_freqsep_…`; 표:
   `pass2/consensus_coverage/stem_consensus_234_sonify.md`)
2. **764**를 Dir 통합 기준선으로 유지; 피아노만 볼 때는 **506**
3. 필요 시 395-only·(506∪395) miss-3 등 추가 보강은 별도 청취 후
4. 527의 2k×21은 보류 유지

**B-2 CQT 판정: 기각.** 세션 5 참조.

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
  # 세션 23 (기존 파일 유지, 신규만 추가)
  전체_rms_plain_클릭.wav
  전체_rms_adaptive_noq1_클릭.wav
  전체_rms_adaptive_클릭.wav
  전체_sf_adaptive_재현_클릭.wav
  vn_rms_adaptive_manifest.json

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
  event_sculpt/                              sculpt pass1
    hpss_percussive.wav   ★ 주 후보 (타건 순간)
    hpss_harmonic.wav / lpc_*.wav / sine_*.wav
    sculpt_manifest.json / sculpt_determinism.json
  event_sculpt/pass2/                        pass2 + 클릭 소니파이
    perc_attack_release_removed.wav  ★ 클릭 후보 (사전 추측 최유력)
    perc_soft_gate.wav               ★ 클릭 후보
    perc_attack_release.wav          ★ 클릭 후보 (링잉 오탐?)
    perc_soft_gate_removed.wav         기각
    lpc_o{12,24,36}_*.wav              + *_클릭 / *_lufs_클릭 / *_k_env_클릭
    perc_attack_release_removed_클릭.wav  1250 peaks
    perc_soft_gate_클릭.wav               1071 peaks
    perc_attack_release_클릭.wav           524 peaks
    pass2_manifest.json / pass2_determinism.json
    pass2_clicks_manifest.json / pass2_clicks_determinism.json
    lpc_clicks_manifest.json / lpc_clicks_determinism.json
    lpc_percept_clicks_manifest.json / lpc_percept_clicks_determinism.json
  event_sculpt/
    lpc_residual_클릭.wav / lpc_synthesis_클릭.wav
    lpc_*_lufs_클릭.wav / lpc_*_k_env_클릭.wav
  event_sculpt/tilt/                         틸트 v2 잠금 + 후속 A/B
    perc_tilt_high.wav / perc_tilt_high_클릭.wav           ★ v2 기준선 (356)
    perc_tilt_softmakeup*_클릭.wav                         v3 기각 (보존)
    perc_tilt_k_env_클릭.wav                               A K-weight (이전 최선→2순위) 340
    perc_tilt_k_env_adaptive_클릭.wav                      ★ k_env소재+SF adaptive 502
    perc_tilt_k_env_material_mono.wav                      소재 mono 참조
    perc_tilt_sine_lowgate_클릭.wav                        B sine lowgate (후순위)
    tilt_material_agreement_on_piano_클릭_p251.wav         ★ 4소재 전원일치
    tilt_material_disagreement_on_piano_클릭_p987.wav      ★ 4소재 불일치
    tilt_material_peak_diff_timestamps.md / .json
    tilt_k_env_adaptive_manifest.json / tilt_k_env_adaptive_determinism.json
    perc_raw_클릭.wav                              untilted 대조
    tilt_manifest.json / tilt_determinism.json
    tilt_percept_manifest.json / tilt_percept_determinism.json
  event_sculpt/pass2/lpc_o12_refine/             ★ 세션 21 채택
    lpc_o12_residual_adaptive_클릭.wav           ★ 현재 최선 383
    lpc_o12_residual_rms_plain_클릭.wav          신뢰 기준선 181
    lpc_o12_residual_*_클릭.wav                  softgate/floor 등
    lpc_o12_refine_manifest.json / lpc_o12_refine_determinism.json
  전체_log1p_raw_클릭.wav       696  log1p (raw)
  ... 외 다수 (블록크기×deburst 조합, intersection, 비교)
```

---

## 2. 세션 이력

### 세션 39 (2026-08-10) — HANDOFF 아카이브

세션 38 시점 HANDOFF를
`Docs/Archive/handoff/HANDOFF_2026-08-10_session38_dir-union764-506-adaptive.md`
로 스냅샷. 라이브 `HANDOFF.md`의 직전 스냅샷 포인터를 갱신.

### 세션 38 (2026-08-10) — Dir 통합본 764 채택 (역할 분담)

**판정** (통합·단독 소니파이 청취):
- **506** = 피아노 사건에 더 특화
- **전체_adaptive_클릭** = 저역 비트 / 다른 구조적 사건에 더 특화
- 현 시점 Dir **통합본 = 764** (`506 ∪ adaptive`)로 결정
- **선결 과제**: stem합의 대비 764 **missed 2** 청취 판정
  (`1:22.039`, `1:32.816` — `stem_consensus_234_sonify.md`)
- stem 합의 234 소니파이 산출:
  `run_stem_consensus_234_sonify.py` → low piano / origmix LUFS,
  vote freqsep·vs764 freqsep·miss2 solo

원곡 bed는 `raw`(무LUFS)와 `lufs`(`load_mono`) 병행 유지.
`origmix_g*`는 raw 별칭.

### 세션 37 (2026-08-10) — 506 vs Dir 전체_adaptive (low piano)

**정의**: 506 vs `전체_adaptive` (102-Dir SuperFlux+peaks_adaptive, n=**679**).
통일 3kHz / freqsep adaptive=3k · **506=5k**. low piano ×0.20.

**실행**: `run_cmp506_vs_dir_adaptive_lowpiano.py --determinism-check`. OK.
공통 421 / 506-only 85 / adaptive-only 258 → union **764**.

| 파일 | |
|------|--|
| `…/cmp506_vs_dirAdaptive_low_g0p20_unified3k_클릭_p764_…wav` | 피아노 low 통일 |
| `…/cmp506_vs_dirAdaptive_low_g0p20_freqsep_클릭_p764_…wav` | 피아노 low freqsep |

**원곡 bed** (`run_cmp506_vs_dir_adaptive_original.py`):
| 파일 | |
|------|--|
| `…_origmix_raw_g{1p00\|0p20}_…` | 원곡 원본 볼륨 (무LUFS) |
| `…_origmix_lufs_g{1p00\|0p20}_…` | `load_mono`/TARGET_LUFS (=전체_adaptive 레벨) |
| `…_origmix_g*_…` | raw 별칭 |

**stem합의 포괄률** (분모 234, 재실행):
| 후보 | n | cov |
|------|--:|----:|
| `union_506_or_dirAdaptive` | 764 | **99.1%** |
| `fusion_kenv_agree_only_506` | 506 | 94.0% |
| `dir_전체_adaptive` | 679 | 92.7% |
| `a2_posdist_rescue_395` | 395 | 82.1% |

→ 세션 38에서 **764 통합본 채택**.

### 세션 36 (2026-08-10) — union miss-3 판정 보류·비치명

**청취/판정** (원문 요지 보존):
- 세 사건(`0:21.142`, `0:21.525`, `1:32.816`)은 왼손 피아노 관련일 수
  있으나, 음 밀집 구간이라 **의미 있는 사건인지는 판단 보류**.
- 오른손 집중 목표에서는 누락이 **치명적이지 않음**.
- 기준선 **506** 유지.

문서 반영: `union506_395_consensus_missed.md` + HANDOFF §0/s4.

### 세션 35 (2026-08-10) — 506 vs 395 역할 소니파이 (low piano)

**정의**: ±30ms 1:1. 공통 3k / 506-only 5k / 395-only 1.5k.
bed = BS piano ×0.20만. 풀볼륨 piano 없음.

**실행**: `run_cmp506_vs_395_lowpiano.py --determinism-check`. OK.
공통 **289** / 506-only **217** / 395-only **106** (이벤트 합 612).

| 파일 | |
|------|--|
| `…/cmp506_vs_395_low_g0p20_freqsep_클릭_p612_c289_6o217_3o106.wav` | 3역할 합 |
| `…/cmp506_vs_395_low_g0p20_unified3k_클릭_p612_c289_6o217_3o106.wav` | 통일 3kHz |
| `…/cmp506_vs_395_low_g0p20_common3k_클릭_p289.wav` | 공통만 |
| `…/cmp506_vs_395_low_g0p20_506only5k_클릭_p217.wav` | 506전용 |
| `…/cmp506_vs_395_low_g0p20_395only1p5k_클릭_p106.wav` | 395전용 |

**stem합의 포괄률** (재실행, 분모 234):
| 후보 | n | cov |
|------|--:|----:|
| `union_506_or_395` | 612 | **98.7%** |
| `fusion_kenv_agree_only_506` | 506 | 94.0% |
| `a2_posdist_rescue_395` | 395 | 82.1% |
| `cmp506_only` | 217 | 17.5% |
| `cmp395_only` | 106 | 5.6% |

**union miss 3** (세션 후속): 문서
`pass2/consensus_coverage/union506_395_consensus_missed.md`
- `0:21.142` · `0:21.525` · `1:32.816`
- 소니파이: `…/union506_395_consensus_missed_low_g0p20_클릭_p3.wav`
- **union+miss3 (615)** low piano:
  - full: `…/union506_395_plus_miss3_low_g0p20_unified3k_클릭_p615.wav`
  - full freqsep (miss=5k): `…_freqsep_클릭_p615_u612_m3.wav`
  - excerpt 21s∥1:32: `…_unified3k_excerpt21_132_클릭_p615.wav` /
    `…_freqsep_excerpt21_132_클릭_p615_u612_m3.wav`

**사용자 판정 (miss 3)**:
- 세 사건은 **왼손 피아노 관련일 수 있으나**, 음 밀집 구간이라
  **정확히 의미 있는 사건인지는 판단 보류**.
- **오른손 피아노에 집중하는 현재 목표**에서는 이 누락이
  **치명적이지 않다**고 판단.
- 따라서 작업 기준선은 계속 **506** (필요 시 진단용으로 union/615 유지).

**현재 상태**: miss-3는 보류·비치명. 506 기준 다음 단계 진행.

### 세션 34 (2026-08-10) — 506 채택 / 527 보류 · 5k×4 수기 로그

**판정**:
- **현재까지의 최선 = 506** (`fusion_kenv_agree_only_*`):
  kenv_adaptive(502) + LPC-order agreement 전용 5k×4. 2k×21 제외.
- **527 보류** (`fusion_kenv_agree_o12db_*`): 2k×21에 유효·비유효 혼재.

**5k×4 청취 규명**: 모두 **유효한 사건 복구**. (freqsep 소니파이로 확인)

**수기 로그 (원문 보존)**:
```
모두 오른 손 타건 사건.
1:13 낮은 음 타건에 반응
1:20 낮은 음 타건에 반응
1:32 주 선율 타건에 반응
1:43 주 선율 타건에 반응
```

**기계 시각 대조** (`agree_only` peak_times, ±30ms 계열):
| 수기 | peak_times_s |
|------|-------------|
| 1:13 | `1:13.517` (73.517s) |
| 1:20 | `1:20.636` (80.636s) |
| 1:32 | `1:32.334` (92.334s) |
| 1:43 | `1:43.141` (103.141s) |

**기준 파일**:
- `…/fusion_kenv_agree_only_on_piano_클릭_p506.wav`
- `…/fusion_kenv_agree_only_on_piano_low_g0p20_클릭_p506.wav`
- `…/fusion_kenv_agree_only_on_piano_freqsep_클릭_p506_5k4.wav`

### 세션 33 (2026-08-10) — stem합의 포괄률 일괄 평가

**가능 여부**: 예. 세션10 잠긴 `stem_consensus_metrics.json`(합의 **234**,
A-2+pos rescue × 3 piano stems, ±30ms 2+)를 분모로 재사용.
검출기 재실행 없음. 스크립트: `eval_consensus_coverage.py`.

**산출**: `pass2/consensus_coverage/sculpt_consensus_coverage.md` (+ `.json`).
395 재현 **82.1%** OK.

| 후보 | peaks | coverage |
|------|------:|---------:|
| `perc_tilt_k_env_adaptive` | 502 | **94.0%** |
| `fusion_kenv_agree_o12db` | 527 | **94.0%** |
| `perc_raw` | 1104 | 85.9% |
| `lpc_o36_sf_adaptive` | 418 | 85.5% |
| `a2_posdist_rescue_395` | 395 | 82.1% |
| `lpc_o12_sf_adaptive` | 383 | 81.2% |
| `lpc_order_agreement` | 325 | 76.9% |
| `perc_tilt_k_env` | 340 | 71.8% |

참고: fusion extras(agree-only 4, o12db 21)의 합의 포괄은 **0** —
fusion이 kenv_ad와 같은 94.0%인 이유.

### 세션 32 (2026-08-10) — kenv∪agree∪o12db fusion + freqsep

**요청**: k_env_adaptive 기반에 LPC-order agreement 적용 + o12 전용에
버스트 대응 후 추가; 클릭 주파수 분리 버전도.

**고정 규칙**: 합치 순서 kenv → agree(±30ms) → o12-only에 chrono **100ms**
wide-gap deburst 후 추가. freqsep: kenv **3k** / agree-only **5k** / o12db **2k**.

**실행**: `run_fusion_kenv_agree_o12_on_piano.py --determinism-check`. OK.

| 단계 | n |
|------|--:|
| kenv | 502 |
| agree-only 추가 | 4 |
| o12 raw extra | 21 |
| o12 deburst drop | 0 (이미 ≥100ms 간격) |
| **unified final** | **527** |

| 파일 | |
|------|--|
| `…/fusion_kenv_agree_o12db_on_piano_클릭_p527.wav` | 통일 3kHz |
| `…/fusion_kenv_agree_o12db_on_piano_freqsep_클릭_p527_5k4_2k21.wav` | 3k×502 / 5k×4 / 2k×21 |
| `…/fusion_kenv_agree_o12db_unifiedL_freqsepR_p527_5k4_2k21.wav` | L=통일 / R=freqsep |
| `…/perc_tilt_k_env_adaptive_nopiano_클릭_p502.wav` | 클릭만 (kenv) |
| `…/lpc_o12_residual_sf_adaptive_nopiano_클릭_p383.wav` | 클릭만 (o12) |
| `…/kenvAd_nopianoL_o12_nopianoR_클릭_p502x383.wav` | L=kenv / R=o12 클릭만 |
| `…/fusion_kenv_agree_o12db_freqsep_nopiano_클릭_p527_5k4_2k21.wav` | freqsep 클릭만 |
| `…/perc_tilt_k_env_adaptive_on_piano_low_g0p20_클릭_p502.wav` | kenv + 피아노×0.20 |
| `…/lpc_o12_…_on_piano_low_g0p20_클릭_p383.wav` | o12 + 피아노×0.20 |
| `…/kenvAd_on_piano_lowL_o12_on_piano_lowR_g0p20_클릭_p502x383.wav` | L/R low piano A/B |
| `…/fusion_…_on_piano_freqsep_low_g0p20_클릭_p527_5k4_2k21.wav` | freqsep + 피아노×0.20 |
| `…/fusion_kenv_agree_only_on_piano_클릭_p506.wav` | 보수: kenv+5k×4 (2k 제외) |
| `…/fusion_kenv_agree_only_on_piano_low_g0p20_클릭_p506.wav` | 보수 low piano |
| `…/fusion_kenv_agree_only_on_piano_freqsep_클릭_p506_5k4.wav` | 보수 freqsep |
| `…/fusion_kenv_agree_only_on_piano_freqsep_low_g0p20_클릭_p506_5k4.wav` | 보수 freqsep low |

### 세션 31 (2026-08-10) — tilt/perc 잠금 소재 multi-way ±30ms on-piano

**정의**: 이미 잠긴 peak_times만 사용(검출기 재튜닝 없음).
포함: `perc_raw`(1104) + `perc_tilt_high`(356) + `perc_tilt_k_env`(340) +
`perc_tilt_k_env_adaptive`(502). ±30ms 클러스터 → **전원일치 251** /
**불일치 987** (=1238). 스킵 없음.

**문서**: `tilt/tilt_material_peak_diff_timestamps.md` (+ `.json`).
생성기: `gen_tilt_material_peak_diff_doc.py`.
공유: `tilt_material_presence_clicks.py`.

**실행**:
`run_tilt_material_agreement_on_piano.py --determinism-check` OK;
`run_tilt_material_disagreement_on_piano.py --determinism-check` OK.
보호 tilt WAV 미변경.

| 파일 | peaks |
|------|------:|
| `…/tilt_material_agreement_on_piano_클릭_p251.wav` | 251 |
| `…/tilt_material_agreement_pianoL_click_pianoR_dry_p251.wav` | 251 |
| `…/tilt_material_disagreement_on_piano_클릭_p987.wav` | 987 |
| `…/tilt_material_disagreement_pianoL_click_pianoR_dry_p987.wav` | 987 |

매니페스트/결정론: `tilt_material_*_on_piano_manifest.json` /
`tilt_material_*_on_piano_determinism.json`.

### 세션 30 (2026-08-10) — LPC order agreement on-piano

**정의**: 세션 29와 동일 ±30ms / 6-order 클러스터; **all-six 일치만** 클릭
(불일치 **171** 제외 → **325**). 공유 모듈: `lpc_order_presence_clicks.py`.

**실행**: `run_lpc_order_agreement_on_piano.py --determinism-check`. OK.

| 파일 | peaks |
|------|------:|
| `…/lpc_order_agreement_on_piano_클릭_p325.wav` | 325 |
| `…/lpc_order_agreement_pianoL_click_pianoR_dry_p325.wav` | 325 (L=클릭, R=dry) |

매니페스트: `lpc_order_agreement_on_piano_manifest.json`.

### 세션 29 (2026-08-10) — LPC order disagreement on-piano

**정의**: o4/o6/o8/o12/o24/o36 SuperFlux+peaks_adaptive를 ±30ms 클러스터링;
**6개 order가 모두 일치하지 않는** 클러스터만 클릭 (all-six **325** 제외 → **171**).

**실행**: `run_lpc_order_disagreement_on_piano.py --determinism-check`. OK.

| 파일 | peaks |
|------|------:|
| `…/lpc_order_disagreement_on_piano_클릭_p171.wav` | 171 |
| `…/lpc_order_disagreement_pianoL_click_pianoR_dry_p171.wav` | 171 (L=클릭, R=dry) |

매니페스트: `lpc_order_disagreement_on_piano_manifest.json`.
피크 소스·클러스터 로직: `gen_lpc_order_peak_diff_doc.py` (`load_series` / `cluster_presence`).

### 세션 28 (2026-08-10) — LPC o4/6/8 + k_env_adaptive on-piano

**피크 차이 문서** (세션 후속): `pass2/lpc_order_peak_diff_timestamps.md`
(±30ms; o4~o36 vs o12 전용 타임스탬프). 생성기: `gen_lpc_order_peak_diff_doc.py`.

**요청**: o12 미만 LPC도 `*_sf_adaptive_on_piano_클릭_*` 방식; k_env_adaptive도 피아노 합성.

**고정 격자**: orders `{4,6,8}` (frame/hop/pre = LPC_PARAMS 불변). pass2 o12/24/36 미변경.

**실행**: `run_lpc_low_and_k_env_on_piano.py --determinism-check`. OK.

| 파일 | peaks |
|------|------:|
| `…/lpc_sf_adaptive_on_piano/lpc_o4_…_on_piano_클릭_p387.wav` | 387 |
| `…/lpc_o6_…_on_piano_클릭_p377.wav` | 377 |
| `…/lpc_o8_…_on_piano_클릭_p371.wav` | 371 |
| `…/perc_tilt_k_env_adaptive_on_piano_클릭_p502.wav` | 502 |

잔여·합성·residual클릭: `pass2/lpc_low_order/`.
참고: o12 on-piano = **383**. o4가 약간 더 많고(387), o6/o8는 다소 적음.

### 세션 27 (2026-08-10) — sf_adaptive × 피아노 스템 청취팩

**실행**: `run_lpc_sf_adaptive_on_piano.py --determinism-check`. OK.
피크 시각 = 각 order의 sf_adaptive(o12=refine adaptive). 피아노 = BS raw stem.
원 residual 클릭 WAV 미변경.

| 파일 | peaks |
|------|------:|
| `…/lpc_sf_adaptive_on_piano/lpc_o12_…_on_piano_클릭_p383.wav` | 383 |
| `…/lpc_o24_…_on_piano_클릭_p406.wav` | 406 |
| `…/lpc_o36_…_on_piano_클릭_p418.wav` | 418 |

각 order별 `pianoL_residClickR_p*` 스테레오도 동봉.

### 세션 26 (2026-08-10) — 클릭 파일명에 피크 수 (D-v2-04)

**규칙**: 신규 `{stem}_클릭_p{N}.wav`. `io_util.click_wav_name`.
기존 일괄 개명 없음. o24/o36·k_env adaptive 러너부터 적용·재산출.

### 세션 25 (2026-08-10) — LPC o24/o36 sf_adaptive 클릭

**실행**: `run_lpc_o24_o36_sf_adaptive.py --determinism-check`. OK.

| 파일 | peaks | vs o12 adaptive ±30ms |
|------|------:|------------------------|
| `pass2/lpc_sf_adaptive/lpc_o24_residual_sf_adaptive_클릭.wav` | 406 | 공통 359 / o12전용 24 / o24전용 47 |
| `pass2/lpc_sf_adaptive/lpc_o36_residual_sf_adaptive_클릭.wav` | 418 | 공통 354 / o12전용 29 / o36전용 64 |

**참고**: k_env(340)↔k_env_adaptive(502)의 274/66/228은
`tilt_k_env_adaptive_manifest.json` 수치만. **비교 클릭 WAV 없음.**
단독 파일: `perc_tilt_k_env_클릭.wav`, `perc_tilt_k_env_adaptive_클릭.wav`.

### 세션 24 (2026-08-10) — perc_tilt_k_env 소재에 SuperFlux adaptive

**VN 종결**: RMS-adaptive 실험 종료. **기존 SuperFlux adaptive 방식 유지.**

**실행**: `run_tilt_k_env_adaptive.py --determinism-check`. OK.
소재 = `perc_tilt_high`(tilt→LUFS) → mono → K-weight (k_env 클릭과 동일).
탐지 = SuperFlux + peaks_adaptive. 기존 tilt WAV **미변경**.

**산출**:
- `tilt/perc_tilt_k_env_adaptive_클릭.wav` — **502** peaks
- `tilt/perc_tilt_k_env_material_mono.wav` — 소재 mono 참조

**관측 (±30ms)**: vs 기존 k_env(340) 공통 274 / k_env전용 66 / adaptive전용 228.
(rms 재현 340 = stored k_env와 일치.)

**현재 상태**: **k_env+adaptive 청취 대기.**

### 세션 23 (2026-08-10) — VN에 RMS/SF adaptive 소니파이 이식

**동기**: Dir residual 실험만으로는 판정 어려움 → 본선곡 VN 전곡에 동일 4변형.

**실행**: `run_vn_rms_adaptive.py --determinism-check`. OK.
기존 `out/sonify/VN/전체_*`(adaptive/gate/norm/Q1/SIR…) **SHA 불변** 검증.

**신규** (`out/sonify/VN/`):

| 파일 | peaks |
|------|------:|
| `전체_rms_plain_클릭.wav` | 3592 |
| `전체_rms_adaptive_noq1_클릭.wav` | 3900 |
| `전체_rms_adaptive_클릭.wav` | 4202 |
| `전체_sf_adaptive_재현_클릭.wav` | 1208 |

±30ms: plain ⊆ noq1 ⊆ rms_adaptive. vs sf: 공통 1020 / sf전용 188 / rms전용 3182.
(하드코어 전곡에서 RMS-adaptive가 SF보다 훨씬 빽빽함 — 청취 판정 필요.)

**현재 상태**: VN RMS-adaptive **기각 — 기존 SF adaptive 유지** (세션 24).

### 세션 22 (2026-08-10) — adaptive에 SuperFlux→RMS(plain) 치환 실험

**동기**: 채택된 peaks_adaptive 골격에서 온셋 함수만 plain RMS로 교체.

**실행**: `run_lpc_o12_rms_adaptive.py --determinism-check`. OK.
입력: `lpc_o12_residual`만.
- env = RMS 2048/256 (rms_plain과 동일 그리드)
- Q1 대역 = ONSET_BANDS Butterworth bandpass → 동일 RMS (SuperFlux bands 아님)
- `rms_adaptive_noq1` = block-gate+norm만 (Q1 제외)

**관측**:

| variant | peaks |
|---------|------:|
| rms_plain | 181 |
| rms_adaptive_noq1 | 253 |
| sf_adaptive (대조=채택) | 383 |
| rms_adaptive | 453 |

±30ms: plain ⊆ noq1 ⊆ rms_adaptive (상위가 하위를 완전히 포함).
rms_adaptive vs sf: 공통 246 / sf전용 137 / rms전용 207.

**현재 상태**: **RMS-adaptive 청취 대기** (sf 채택본과 비교).

### 세션 21 (2026-08-10) — lpc_o12 refine: 강화 floor / softgate / adaptive

**동기**: 상향 open계에서 사건 아닌 지점 클릭 지배. o12 residual만 재가공.

**실행**: `run_lpc_o12_refine_clicks.py --determinism-check`. OK.

| variant | peaks | 요지 |
|---------|------:|------|
| rms_plain | 181 | 대조 |
| softgate | **162** | soft_env_gate 후 RMS+Otsu (가장 적음) |
| up_g4_floor_p50 | 306 | 바닥 p50 강화 |
| up_g4_floor_rel25 | 294 | e&lt;0.25·T 제외 |
| up_g4_floor_p25 | 356 | |
| up_g4_floor_rel10 | 361 | |
| up_g4_softgate | 351 | 상향 후 softgate |
| adaptive | 383 | SuperFlux+peaks_adaptive (=전체_adaptive) |
| softgate_adaptive | 370 | |
| up_g4_floor_p25_adaptive | 412 | |

**사용자 청취 판정**:
- **`rms_plain`**: 신뢰할 만함 (기준선으로 유지).
- **`adaptive`**: **가장 우수 — 현재 s4 최선.** 직전 최선
  `perc_tilt_k_env_클릭`보다 나은 수준.
- 상향·강화 floor·softgate 조합은 주선이 아님(비사건/이득 불명확 쪽).

**현재 상태**: lpc_o12+adaptive **채택**. 다음 단계(누락·교차 확인) 대기.

### 세션 20 (2026-08-10) — LPC residual 상향 레벨러 다중 클릭

**동기**: 약한 residual을 강한 구간과 동급으로. 센 구간은 g=1 유지.
미약 스템이므로 **tiny-e 바닥 제외를 하지 않은 버전을 반드시 포함**.

**규칙**: RMS→블록 2s p99 = T. `g=clip(T/max(e,eps), 1, g_max)`.
소프트게이트 없음. 기록 hard clip.

| variant | g_max | floor | otsu_split |
|---------|------:|-------|------------|
| up_open | ∞ | 없음 | no |
| up_g4 | 4 | 없음 | no |
| up_g10 | 10 | 없음 | no |
| up_open_p01 | ∞ | 블록 p1 미만 g=1 | no |
| up_open_p05 | ∞ | 블록 p5 미만 g=1 | no |
| up_otsu_open | ∞ | 없음 | e≤Otsu만 부스트 |
| up_otsu_g10 | 10 | 없음 | e≤Otsu만 부스트 |

**실행**: `run_lpc_upward_clicks.py --determinism-check`. OK.

**관측 (o24 residual 클릭 peaks; plain=179)**:
open 526 / g4 335 / g10 424 / p05 495 / p01 519 / otsu_open 552 / otsu_g10 486.
open계 g_max_applied ≈ 10³–10³ 급 → 클립 다수 가능(청취 주의).

**현재 상태**: 상향 변형은 비사건 클릭 문제로 **후순위** (세션 21 판정).

### 세션 19 (2026-08-10) — LPC LUFS / K-env 클릭 (softgate 보류)

**동기**: LPC에도 음량·지각 보정 버전 클릭. softgate는 감쇠 상시 ON 이슈로 보류.
ISO 226 등청감은 SPL 자유도 → 미사용. K-weight = BS.1770-4 표준 계수.

**실행**: `run_lpc_percept_clicks.py --determinism-check`.
- `lufs`: LUFS−23 → RMS+Otsu → 클릭 (clip)
- `k_env`: LUFS−23 → K-weight → RMS+Otsu → 클릭
피크픽은 plain LPC 클릭과 동일(2s-p99 없음).

**관측**:
- residual: lufs peaks = plain (179/181/176) — 전역 이득만이라 Otsu 집합 불변 예상과 일치.
  k_env도 residual에서 거의 동일(o12만 181→180).
- synthesis: lufs=327(=plain); **k_env=287** (전 order 동일 수) — 지각 필터가
  synthesis 쪽에서만 유의미한 집합 변화.

**현재 상태**: **LPC plain / lufs / k_env 클릭 청취 대기.**

### 세션 18 (2026-08-10) — LPC 전 산출 클릭 소니파이

**동기**: perc 기반이 아닐 때(LPC residual/synthesis)를 클릭으로 들어보기.

**실행**: `run_lpc_clicks.py --determinism-check`.
규칙 = pass2_clicks와 동일 (RMS+Otsu+30ms, 3kHz). 395/L/R 없음.

**관측 (peaks)**:

| 파일 | peaks |
|------|------:|
| `lpc_residual_클릭` (=o24) | 179 |
| `lpc_o12_residual_클릭` | 181 |
| `lpc_o24_residual_클릭` | 179 |
| `lpc_o36_residual_클릭` | 176 |
| `lpc_*_synthesis_클릭` (전 order) | **327** (전부 동일 수) |

WAV·peak_times 재실행 일치.

**현재 상태**: **LPC 클릭 청취 대기** (틸트 A/B와 병행).

### 세션 17 (2026-08-10) — 틸트 후속: K-env + sine_lowgate 설계·산출

**판정**: v3 softmakeup **기각** — 사용자: 이전(v2)과 동일 결과, v2가 나음.
v2를 기준선으로 잠금.

**설계(D-21)**:
- A `k_env`: 틸트+LUFS 파형 유지, RMS 직전 BS.1770-4 K-weight만 (표준 계수, 새 파라미터 없음).
- B `sine_lowgate`: pass1 `sine_residual` STFT `f < f_ref(1kHz)` → RMS→2s-p99 soft,
  `env_det = env_v2_n * (1-soft)` → Otsu. λ 없음.
- A∘B는 1차 청취 후. ISO 226 등청감은 제외(기준 SPL 자유도).

**실행**: `run_tilt_percept.py --determinism-check`.
WAV·peak_times 재실행 일치.

**관측**:
- v2 재현 356 peaks (잠금과 일치)
- A k_env: **340** peaks; vs v2 ±30ms 공통 337 / v2전용 19 / A전용 3
- B sine_lowgate: **346** peaks; 공통 238 / v2전용 118 / B전용 108
- gate: on_frac≈1.0 (soft>0), mean≈0.623 (전 프레임 감쇠 강도)

**현재 상태**: **v2 → A → B 청취 대기.**

### 세션 16 (2026-08-10) — percussive 스펙트럼 틸트 + 클릭 (v2 + v3 softmakeup)

**동기**: 클릭 왼손(저역) 편향. 단순 저↓고↑ 틸트(사용자 아이디어와 독립).

**v1**: 틸트 후 soft-scale 기록 → 청취 음량 붕괴, peaks=140 (raw 1177).
사용자: 음량 낮아 클릭 미생성 다수.

**v2** (`run_tilt_clicks.py`):
- 틸트 (α=1, f_ref=1kHz, f_floor=80) → LUFS −23 → 피크·소니파이
- RMS env 2s p99 → Otsu; 기록 **hard clip**
- 관측: tilt 클릭 **356** peaks; raw 대조 1104

**v3 softmakeup** (사용자 요청):
- 틸트 → **soft_scale(0.98)** → LUFS makeup(−23) → 피크·소니파이(clip)
- 산출: `perc_tilt_softmakeup.wav` / `perc_tilt_softmakeup_클릭.wav`
- 관측: peaks=**356** (=v2), 피크 시각 동일, 결정성 OK. soft→makeup이 전역 이득에 가깝게 역전되어 v2와 거의 동치.

**사용자 아이디어 (미실행, 다음 후보)**:
1. LUFS류 **지각 대역 주목도**(등청감 / K-weighting) 기반 음량 보정
2. **`sine_residual`**로 저역 과다 겨냥 보정

**현재 상태**: v2 **기준선 잠금**. v3 **기각** (세션 17). 후속 A/B는 세션 17.

### 세션 15 (2026-08-10) — pass2 3종 모노 클릭 소니파이

**실행**: `run_pass2_clicks.py --determinism-check`.
입력: pass2의 attack_release_removed / soft_gate / attack_release.
방식: 잔여 stereo→mono mean, RMS env(2048/256) + Otsu + greedy 30ms,
클릭 3kHz overlay (Dir식). 395 비교·L/R 없음.

**관측**:
- removed_클릭: 1250 peaks
- soft_gate_클릭: 1071 peaks
- attack_release_클릭: 524 peaks
- WAV·peak_times 재실행 일치

**현재 상태**: 산출 완료. **사용자 클릭 청취 판정 대기.**
피크 수·임계 역튜닝 없음.

### 세션 14 (2026-08-10) — percussive 점진 축소 + LPC order 스윕 + 청취

**실행**: `run_pass2.py --determinism-check`. 입력 hpss_percussive + BS piano.
WAV SHA 재실행 일치. 클릭/계수 없음(판정 후 다음 단계).

**고정 규칙**:
- A1 soft_env_gate: RMS 2048/256, 2s p99 norm, Otsu soft mask 곱
- A2 attack_release: att=5ms, rel=40ms (hop-grid follower), / 2s p99 곱
- LPC order ∈ {12, 24, 36}; frame=2048, hop=512, pre=0.97 고정

**관측(수치)**:
- soft_gate: peak=0.536 rms=0.036; removed peak=0.054 rms=0.004; Otsu thr=0.479
- attack_release: peak=0.514 rms=0.026; removed peak=0.144 rms=0.014; gain_mean=0.515
- lpc residual rms: o12=0.0102, o24=0.0090, o36=0.0081
- lpc_o24_residual SHA = pass1 `lpc_residual`과 동일 (재현)

**사용자 청취 판정**:
- `perc_soft_gate_removed`: **기각** — 불안정하고 누락이 큼. 청취상 LPC 계열이
  이 역할(제거/이산 잔여)을 대신한다.
- LPC o12/24/36: order가 커질수록 **더 이산화**되는 질적 경향. 누락 케이스는
  들리지 않음. 현 그리드로는 **보조 활용 또는 제외**가 맞음. 더 많은 사건이
  포함되도록 만들 수 있으면 개선 여지.
- **클릭 소니파이 가치 있음** (3종):
  - `perc_attack_release_removed` — 듣기 전 추측 **가장 뛰어날 수 있음**
  - `perc_soft_gate` — 좋음
  - `perc_attack_release` — 링잉 오탐이 약간 있을 수도

**해석**: pass2의 본선은 soft_gate / attack_release / attack_release_removed
세 WAV의 클릭 표현으로 이어진다. soft_gate_removed와 현 LPC 스윕은 주선에서
내린다(LPC는 포괄↑ 개선 시에만 재검토). 파라미터·그리드 역튜닝 없음.

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
| **HPSS percussive (sculpt)** | BS piano → anisotropic median P | 타건 순간 포괄; sculpt 초기 주 후보 | LPC o12 경로로 주선 이관 |
| **lpc_o12 + adaptive** | o12 residual → SuperFlux+peaks_adaptive | **현재 s4 최선** (383); tilt k_env 상회 | 누락·교차 확인 여지 |
| **lpc_o12 rms_plain** | o12 residual → RMS+Otsu | **신뢰 가능 기준선** (181) | 포괄은 adaptive보다 낮음 |

### 보조 유효

| 기법 | 원리 | 역할 |
|------|------|------|
| **SIR(u3)** | 분광 균일도 ch_min/ch_max + 2-of-3 동시극대 | cry 과다탐지 억제, Q1 병용 |
| **A-2 sliding A-3** | block p99 → sliding p99 | 확실한 피아노 사건 중심, 최종 용도 보류 |
| **B-3 tri-complex** | novelty×flux×complex 연속값 3-way | 피아노 이벤트 추종 유망, 주/보조 용도 보류 |
| **A2∩complex core** | ±30ms 일대일 공통 사건 | 타건 누락 큰 정밀도 보조 |
| **positive-distribution rescue** | A-2와 비매칭인 positive×flux 사건 추가 | A-2 보존 + 실제 타건 40개 복구 |
| **BS-Roformer piano stem** | 동일 원음 6-stem 분리 | 주 귀속 참조; Spleeter/Demucs로 감사 |
| **perc_soft_gate (sculpt)** | P × Otsu soft env mask | 클릭 후보 (병행) |
| **perc_attack_release (sculpt)** | P × att/rel gain | 클릭 후보 (링잉 오탐?) |
| **perc_attack_release_removed** | P × (1−gain) | 클릭 후보 |
| **tilt v2 / k_env (A)** | 틸트→LUFS; (+K-weight env) | **이전 최선→2순위** (k_env 340) |
| **LPC synthesis (sculpt)** | LPC 재합성 | 사건↔링잉 대비, 보조 청취 |
| **HPSS harmonic (sculpt)** | H 성분 | 전자피아노; 가벼운 전처리 후보 |

### 불충분 (s4 피아노)

| 기법 | 판정 |
|------|------|
| A-1 WTMM-inspired chain | 대조 대부분 유지+87개 추가, artifact 분리 이득 불명확 |
| A-1+A-3 결합 | 일부 개선과 burst가 혼재해 단독 채택 불충분 |
| A2-only residual | 실제 사건/artifact 여부와 감쇠 안전성 불명 |
| complex-only rescue | 유효 사건과 비피아노 burst 혼재, 선별 필요 |
| sine_residual (sculpt) | 울림·링잉 큼; 중저역 게이트 입력으로만 재사용 |
| sine_tonal (sculpt) | 배음 포락선으로 들림; 본 프로젝트 사건 축은 의문 |
| perc_soft_gate_removed | 불안정·누락 큼; LPC가 역할 대체 → 기각 |
| tilt sine_lowgate (B) | 감쇠 상시 ON 이슈; 후순위 |
| LPC upward open 계열 | 비사건 지점 클릭 지배 → 후순위 |
| o12 강화 floor / up+softgate | adaptive·rms_plain 대비 주선 아님 |

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
| tilt v3 softmakeup | peaks·시각=v2와 동일; soft→LUFS≈전역이득 역전; v2 선호 |

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
  stem_event_sculpt/    — tilt v2 잠금; A/B percept 러너 청취 대기
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
| D-v2-04 | **신규 클릭 WAV 파일명에 피크 수 포함**: `{stem}_클릭_p{N}.wav` (`io_util.click_wav_name`). 기존 파일 일괄 개명 금지. 매니페스트 `n_peaks`와 `p{N}` 불일치 = 버그. |

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
