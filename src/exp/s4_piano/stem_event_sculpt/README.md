# s4 piano stem event sculpt

BS-Roformer piano stem을 **전처리 → 전장 WAV 청취**로 이산 사건 잔여
후보를 만드는 실험 작업공간이다.

**Dir 통합본(764) · 전체_adaptive · 506 소개**:
[`Docs/dir_764_pipeline.md`](../../../../Docs/dir_764_pipeline.md)

기존 `_onset_*.py` / `stem_validation` / `transcription`과 분리한다.
395 대비 A/B 소니파이·L/R stem 진단 포맷·클릭 오버레이는 만들지 않는다.
산출 WAV 자체가 청취물이다.

## 클릭 파일명 (D-v2-04, 2026-08-10~)

신규 클릭 WAV: **`{stem}_클릭_p{N}.wav`** (`io_util.click_wav_name`).
예: `lpc_o24_residual_sf_adaptive_클릭_p406.wav`.
기존 파일은 개명하지 않는다. `n_peaks` ≠ `p{N}`이면 버그.

## 목표

- 귀납·저누락: 검출기 파라미터 고르기보다 성분 분리 잔여를 듣는다.
- 1차: HPSS / LPC / sinusoidal 병행 산출 → 청취 판정 (세션 13)
- 2차: **hpss_percussive 점진 축소** + **LPC order 스윕** (세션 14)
- 클릭/계수는 이후 단계

## Pass1 — 고정 파라미터

| 패스 | 사건 후보 | 보완 | 값 |
|------|-----------|------|-----|
| HPSS | `hpss_percussive.wav` | `hpss_harmonic.wav` | kernel=31, power=2, margin=1, n_fft=2048, hop=256 |
| LPC | `lpc_residual.wav` | `lpc_synthesis.wav` | order=24, frame=2048, hop=512, pre-emphasis=0.97 |
| Sinusoidal | `sine_residual.wav` | `sine_tonal.wav` | local-max ∧ mag≥frame p90 |

## Pass2 — D-21 고정

입력: `hpss_percussive.wav`(본선), `bs_roformer/piano.wav`(LPC 스윕).

| 산출 | 규칙 |
|------|------|
| `perc_soft_gate` | RMS 2048/256 → 2s p99 norm → Otsu soft mask `clip(env/thr,0,1)` 곱 |
| `perc_attack_release` | 동일 RMS → attack 5ms / release 40ms 추종 → `/` 2s p99 → 곱 |
| LPC order | **{12, 24, 36}**만. frame/hop/pre-emphasis 고정. 청취 후 그리드 점 추가 금지 |

제거분(`*_removed`)도 함께 산출. soft-limit 0.98은 파일 단위만.

## 출력

```
out/stems/Dir/event_sculpt/          # pass1
  hpss_*.wav, lpc_*.wav, sine_*.wav, sculpt_manifest.json
out/stems/Dir/event_sculpt/pass2/    # pass2
  perc_soft_gate.wav / perc_soft_gate_removed.wav
  perc_attack_release.wav / perc_attack_release_removed.wav
  lpc_o{12,24,36}_residual.wav / lpc_o{12,24,36}_synthesis.wav
  pass2_manifest.json / pass2_determinism.json
```

## 실행

```powershell
$py = "E:\game\Music Hermeneutic AI\.venv\Scripts\python.exe"
& $py "src\exp\s4_piano\stem_event_sculpt\run_passes.py"
& $py "src\exp\s4_piano\stem_event_sculpt\run_pass2.py" --determinism-check
```

## 청취 순서 (pass2)

1. `hpss_percussive` → `perc_soft_gate` ↔ removed → `perc_attack_release` ↔ removed
2. `lpc_o12/24/36_residual` (synthesis는 편차·누락 해석 보조)

## 청취 판정 (세션 13, pass1)

| 산출 | 판정 |
|------|------|
| **hpss_percussive** | **주 후보** |
| **lpc_residual** | 보조(조건부) → pass2 order 스윕 |
| **lpc_synthesis** | 유효 |
| hpss_harmonic / sine_* | 전처리·배음 축 (본선 사건과 거리)

## 청취 판정 (세션 14, pass2)

| 산출 | 판정 |
|------|------|
| **perc_attack_release_removed** | **클릭 후보** (사전 추측 최유력) |
| **perc_soft_gate** | **클릭 후보** |
| **perc_attack_release** | **클릭 후보** (링잉 오탐?) |
| perc_soft_gate_removed | **기각** — 불안정·누락; LPC가 대체 |
| LPC o12/24/36 | order↑ 이산화(질적), 누락 안 들림 → 보조/제외. 포괄↑ 여지 |

다음: 위 클릭 후보 3종에 이전 Dir 모노 클릭 소니파이. 그리드/파라미터 역튜닝 없음.

## 클릭 소니파이 (세션 15)

```powershell
& $py "src\exp\s4_piano\stem_event_sculpt\run_pass2_clicks.py" --determinism-check
```

| 파일 | peaks |
|------|------:|
| `perc_attack_release_removed_클릭.wav` | 1250 |
| `perc_soft_gate_클릭.wav` | 1071 |
| `perc_attack_release_클릭.wav` | 524 |

고정: RMS+Otsu+30ms, 클릭 3kHz, 잔여 mono 위 오버레이.

## LPC 클릭 소니파이 (세션 18)

perc 비기반 경로. pass1 o24 + pass2 o12/24/36 residual·synthesis 전부.

```powershell
& $py "src\exp\s4_piano\stem_event_sculpt\run_lpc_clicks.py" --determinism-check
```

| 파일 | peaks |
|------|------:|
| `lpc_residual_클릭.wav` / `lpc_o24_residual_클릭.wav` | 179 |
| `lpc_o12_residual_클릭.wav` | 181 |
| `lpc_o36_residual_클릭.wav` | 176 |
| `lpc_*_synthesis_클릭.wav` (전 order) | 327 |

## LPC 지각/음량 클릭 (세션 19)

softgate 보류. ISO 226 미사용.

```powershell
& $py "src\exp\s4_piano\stem_event_sculpt\run_lpc_percept_clicks.py" --determinism-check
```

| 변형 | residual peaks | synthesis peaks |
|------|---------------:|----------------:|
| plain (세션 18) | 179 / 181 / 176 | 327 |
| `*_lufs_클릭` | 동일 (전역이득) | 327 |
| `*_k_env_클릭` | ≈동일 (o12:180) | **287** |

## LPC residual 상향 레벨러 (세션 20)

약한 프레임만 블록 p99로 상향. 센 프레임 g=1. softgate 없음.
**바닥 제외 없는 open 포함** (미약 스템 배려).

```powershell
& $py "src\exp\s4_piano\stem_event_sculpt\run_lpc_upward_clicks.py" --determinism-check
```

출력: `out/stems/Dir/event_sculpt/pass2/lpc_upward/`
예: `lpc_o24_residual_up_g4_클릭.wav`

| variant (o24) | peaks | 비고 |
|---------------|------:|------|
| plain | 179 | 세션 18 |
| up_g4 | 335 | 상한 4× |
| up_g10 | 424 | 상한 10× |
| up_open | 526 | 바닥 제외 없음 (이득 매우 큼→클립) |
| up_open_p01 | 519 | 블록 p1 바닥 |
| up_open_p05 | 495 | 블록 p5 바닥 |
| up_otsu_open | 552 | Otsu 이하만 부스트 |
| up_otsu_g10 | 486 | Otsu + g≤10 |

청취 권장: **g4 → g10 → open → p01/p05 → otsu**.

## lpc_o12 refine (세션 21)

입력: `pass2/lpc_o12_residual.wav`만. 비사건 클릭 억제 목적.

```powershell
& $py "src\exp\s4_piano\stem_event_sculpt\run_lpc_o12_refine_clicks.py" --determinism-check
```

출력: `out/stems/Dir/event_sculpt/pass2/lpc_o12_refine/`

| variant | peaks |
|---------|------:|
| softgate | 162 |
| rms_plain | 181 |
| up_g4_floor_rel25 | 294 |
| up_g4_floor_p50 | 306 |
| up_g4_softgate | 351 |
| up_g4_floor_p25 | 356 |
| up_g4_floor_rel10 | 361 |
| softgate_adaptive | 370 |
| adaptive | 383 |
| up_g4_floor_p25_adaptive | 412 |

`adaptive*` = 본선 `전체_adaptive_클릭`과 동일 SuperFlux+peaks_adaptive.

**청취 판정 (세션 21)**: `rms_plain` 신뢰 가능; **`adaptive` 채택(현재 최선)**,
이전 `perc_tilt_k_env`보다 우수.

## RMS→adaptive 치환 (세션 22)

SuperFlux 대신 plain RMS로 peaks_adaptive 실행.

```powershell
& $py "src\exp\s4_piano\stem_event_sculpt\run_lpc_o12_rms_adaptive.py" --determinism-check
```

출력: `out/stems/Dir/event_sculpt/pass2/lpc_o12_rms_adaptive/`

| variant | peaks | 요지 |
|---------|------:|------|
| rms_plain | 181 | Otsu만 |
| rms_adaptive_noq1 | 253 | gate+norm, Q1 없음 |
| sf_adaptive | 383 | SuperFlux 채택 대조 |
| rms_adaptive | 453 | RMS+full adaptive |

## 틸트 전처리

공통: α=1 틸트 → LUFS −23 → env 2s-p99 → Otsu → **hard clip** 기록.

| 버전 | 파이프 | 판정 |
|------|--------|------|
| **v2** | tilt → LUFS | **기준선 잠금** (`tilt/perc_tilt_high*_클릭.wav`, 356) |
| v3 | tilt → soft_scale → LUFS makeup | **기각** (peaks·시각=v2와 동일) |
| 대조 | untilted LUFS | `tilt/perc_raw_클릭.wav` |

### 틸트 후속 — 지각 엔벨로프 + sine 저역 게이트

| 후보 | 규칙 | 파일 |
|------|------|------|
| A `k_env` | v2 모노 → BS.1770-4 K-weight → RMS | `tilt/perc_tilt_k_env_클릭.wav` |
| B `sine_lowgate` | sine_residual STFT `f < f_ref` → 2s-p99 soft × v2 env | `tilt/perc_tilt_sine_lowgate_클릭.wav` |

A∘B는 1차 청취 후. 새 자유 파라미터 없음 (K=표준, 컷오프=`TILT_PARAMS.f_ref_hz`).

```powershell
$py = "E:\game\Music Hermeneutic AI\.venv\Scripts\python.exe"
& $py "src\exp\s4_piano\stem_event_sculpt\run_tilt_clicks.py" --determinism-check
& $py "src\exp\s4_piano\stem_event_sculpt\run_tilt_percept.py" --determinism-check
```

청취 순서: **v2 → A → B**.

## Workspace

`runtime/`, `work/`는 이 디렉터리 아래 격리·gitignore.
