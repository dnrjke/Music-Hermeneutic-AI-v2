# via_764 — 506 onset 골격 → 피치 채움 MIDI

상위: [`../README.md`](../README.md)  
계획: [`Docs/piano_midi_via_dir764_plan_cursor_grok_4.5.md`](../../../../Docs/piano_midi_via_dir764_plan_cursor_grok_4.5.md)

**한 줄**: 506 타건 시각을 고정하고 스템에서 피치·길이·벨로시티만 채운다 (종단간 AMT 아님).

## 경계 (잠금)

| | 규칙 |
|--|------|
| 형제 | `clean_amt/` · `midi_fuse/` · `stem_norm/` · `event_pitch/` **import·out 공유 금지** |
| s4 | **import 금지**. 피크 JSON·piano WAV는 **읽기 전용** |
| venv | 새 venv 없음 → `../clean_amt/env/.venv` 런타임만 |
| onset | **P0 = 506 only** (`conservative_kenv_agree_only`) |
| 764/adaptive | 점검·소니파이 참고만 (D0 비사용) |

## 마일스톤

| 단계 | 상태 |
|------|------|
| **D0** | **완료** — 506 → placeholder mid + lowpiano 5k 클릭 (`out/20260812_via764_D0_dir_506/`) |
| **D1** | pyin·harmonic·spectral·fuse·local AMT **전부 청취 no-go 잠금** |
| D2–D5 | **중단** — 피치 후속 `event_pitch` E1–E7도 **전량 no-go**. via_764는 onset 골격·소니파이 자산만 유지 |

## 입력 (RO)

- `out/stems/Dir/event_sculpt/pass2/lpc_sf_adaptive_on_piano/fusion_kenv_agree_o12db_on_piano_manifest.json`
- `out/stems/Dir/bs_roformer/piano.wav`

## 실행

```powershell
$py = "src\exp\s5_midi\clean_amt\env\.venv\Scripts\python.exe"
# D0
& $py src\exp\s5_midi\via_764\scripts\d0_onset_midi.py --config src\exp\s5_midi\via_764\configs\d0_dir.yaml
& $py src\exp\s5_midi\via_764\scripts\sonify_d0_lowpiano.py
# D1
& $py src\exp\s5_midi\via_764\scripts\d1_pitch_fill.py --config src\exp\s5_midi\via_764\configs\d1_dir_pilot.yaml
```

D1 청취: `piano_from_506.mid` vs piano stem (롤에서 멜로디 윤곽). onset은 D0 클릭과 동일 골격.

### D1 청취 판정 (2026-08-12)

- **피치 no-go**: 원곡/스템과 맞지 않음.
- **사건감**: “스템에서 의도한 SuperFlux와 안 맞는다”는 인상.

**진단 (정량)**:

| 항목 | 결과 |
|------|------|
| 피치 vs clip⊕AMT (동구간, ±50ms) | 매칭 124/127 · **동일 pitch 8** · med Δ≈**−27 st** (약 2옥↓) → pyin top-1 **실패** |
| 시각 vs clip AMT onset (±50ms) | 506→clip **97.6%** hit (114/127 @30ms) → **onset 골격은 대체로 맞음** |
| 506 검출 소재 | **원곡/피아노 SuperFlux가 아님**. HPSS perc→tilt→LUFS→K-weight 위 SF-adaptive (+LPC agree×4). |

→ 피치 문제는 실재. “SuperFlux 불일치” 인상은 (1) 잘못된 피치로 인한 청감 어긋남 + (2) 506이 **스템 raw SF가 아닌 tilt/K-env 소재 SF**인 점 + (3) clip AMT보다 성긴 127 vs ~223 onset이 겹친 결과로 설명 가능.

**다음 (D1 재시도 후보, 미착수)**: pyin 폐기 → 창 안 스펙트럼 peak / multipitch top-1·K, 또는 짧은 국소 AMT. onset은 D0/506 유지한 채 피치만 교체.

### D1v2 (2026-08-12) — pyin 대체

onset=506 유지. 방법만 교체. clip AMT 대비 (±50ms, 참고용):

| method | same pitch | med Δst | \|Δ\|mean | pitch med | 산출 |
|--------|----------:|--------:|----------:|----------:|------|
| pyin (legacy) | 8/124 | −27 | 27.3 | 38 | `…_top1_t30_60` |
| **harmonic_peak** | 24/124 | −13 | 19.2 | 54 | `…_harmonic_peak_t30_60/` |
| **spectral_peak** | **28/124** | **−5** | **15.4** | 68 | `…_spectral_peak_t30_60/` |

정량상 spectral이 낫다. 둘 다 mid 청취 후 go/no-go.

### D1v2 청취 판정 (2026-08-12)

- **둘 다 크게 문제** — 주 선율 사건에 **대응하는 시각**이지만 **해당 피치에 대응 못함** → 음율로 들리지 않음.
- spectral이 약간 나을 수 있으나 **미달 · no-go**.
- 해석: 단음 STFT/하모닉 합산은 폴리포니·울림 스템에서 배음/다른 성부를 잡아 멜로디 F0를 놓침. onset 고정 이득은 유지되나 **피치 모듈이 병목**.

### D1c (2026-08-12) — 국소 AMT + fuse 피치 이식

onset=506 유지. 피치만 교체. clip AMT 대비 (±50ms 매칭 124/127):

| method | same pitch | med Δst | \|Δ\|mean | miss | pitch med | 산출 |
|--------|----------:|--------:|----------:|-----:|----------:|------|
| spectral_peak (ref) | 28/124 (23%) | −5 | 15.4 | — | 68 | `…_spectral_peak_t30_60/` |
| **fuse_transplant** | **62/124 (50%)** | **0** | **8.7** | 1 | 73 | `…_fuse_pitch_t30_60/` |
| **local_amt** | 50/124 (40%) | 0 | 10.2 | 1 | 69 | `…_local_amt_t30_60/` |

정량: fuse ≫ local_amt ≫ spectral. fuse는 clip⊕harmonic notes RO 이식 (`max_vel`, ±50ms). local_amt는 창 concat→1회 Transkun→창 매핑 (raw AMT notes 475).

```powershell
& $py src\exp\s5_midi\via_764\scripts\d1_pitch_from_fuse.py --config src\exp\s5_midi\via_764\configs\d1_fuse_pitch_pilot.yaml
& $py src\exp\s5_midi\via_764\scripts\d1_pitch_from_amt_local.py --config src\exp\s5_midi\via_764\configs\d1_local_amt_pilot.yaml
```

### D1c 청취 판정 (2026-08-12) — **no-go 잠금**

- **fuse 이식 실패**: clip⊕harmonic에 해당 506 시점 사건이 없으면 참조 피치가 없음 → 잘못된 피치가 남음. 원리적으로 이식은 764 속행에 부적합.
- **local AMT**도 음율 미달.
- **피치 후속**: [`../event_pitch/`](../event_pitch/) — 같은 \(t_i\)에서 **주파수를 접지 않은** 재추정 (SuperFlux·fuse와 별 축).
