# event_pitch — 주파수 보존 / pitch-wise onset

상위: [`../README.md`](../README.md)  
계획: [`Docs/piano_event_pitch_plan_cursor_grok_4.5.md`](../../../../Docs/piano_event_pitch_plan_cursor_grok_4.5.md)

## 가설 재진술 (청감 · 2026-08-12)

사용자 청감과의 정합:

1. 소니파이에서 **선율 사건으로 들은 것**과 **506 클릭**이 유사했다 → 506은 시각 골격으로 유의미.
2. 506이 기술하는 것은 **어택 변화**(피아노 연관 추측 가능)이지, **선율 음정 값**이 아니다.
3. 목표: 그 어택 시각에서 피치를 **별도 추정**해, 클릭↔청감 매칭이 **피치 차원에서도** 긍정되게 하는 것.
4. E1–E7은 (3)의 피치 추정에 실패 → “506이 쓸모없다”가 아니라 **어택 시각 조건 피치 모듈**이 미달.

축 no-go는 “506→피치 자동 유도” 구현 실패이지, 506 시각 골격의 청감 기각이 아니다.


## 경계

| | 규칙 |
|--|------|
| 형제 | import·out 공유 금지 |
| s4 | import 금지 · 피크/WAV RO |
| venv | `../clean_amt/env/.venv` |

## E10 — 506 **피치 포함 평가** (fuse RO × 마스크) · **약한 부분 신호**

**의의**: go 잠금 `clip⊕harmonic` 피치로 506을 **처음** 피치 차원에서 들어볼 수단.  
**한계**: fuse에 없는 506은 공백 — “순수 506 성능” 전수 평가는 아님.

### 청취 판정 (2026-08-12)

- **설득력 있다(go)까지는 아님.**
- 대략 **반**은 “그때 친 음”, **반**은 다른 음정·성부가 섞임.
- 원본 선율이 **드물게** 비치는 수준 → **약한 부분 신호** (축 go 아님 · 완전 무효도 아님).

해석: 506 시각 골격은 여전히 쓸모 있으나, fuse±tol top-1 픽은 폴리포니에서 선율 F0를 안정히 못 고름. 교집합·귀속 한계와 맞물림.

### 원래 직관 → 남은 가설 (2026-08-12)

특정 순간의 vel/flux·피치 분석 시도들은 결국 **“506을 만든 길과 같은 가족으로 피치도”** 였다.

| | onset (506) | pitch (막힌 칸) |
|--|-------------|-----------------|
| 목표 | 어택 **시각** | 그 시각의 **음정** |
| 나이브 | 저역 등 오탐 | 동일 — bass/울림에 끌림 |
| 506이 한 일 | 소재 가공(perc→tilt→K) 후 SF | — |
| 피치에 옮기면 | 주파수를 접거나 왜곡하면 **음정이 죽음** | **주파수 왜곡 없는** 분리·비교가 맞음 |

E11(레이어 합의) **no-go** 후 → E12(독립 교차·옥타브).

## E11 — 분리 레이어 비교 피치 @ 506 · **no-go**

**청취 판정 (2026-08-12)**: rescue가 유의미한 수준이 못 됨. **피치 추정 오류 → 선율 형성 실패.**

정량 합의(~97%)는 높았으나 무의미 — piano/harmonic/synthesis가 **같은 CQT 가족**이라  
v1 [L-49]/§11-b 교훈과 같음: *상관된 추정기가 일치해도 참이 아니다* (BPM 외부 DB 전부 반속과 동형).

| 창 | n | agree | rescue |
|----|--:|------:|-------:|
| 30–60s | 127 | 124 | 1 |
| 60–90s | 138 | 134 | 138 |

산출: `out/20260812_event_pitch_E11_dir_506_layer_compare_t{30_60,60_90}/`


## E12 — 독립 추정기 교차 + 옥타브 보정 · **no-go**

**청취 판정 (2026-08-12)**: 음·피치 모두 안 맞음 → **실패.**

실패 해석: CQTΔ/pyin/옥타브 보정은 **추정기를 가정·교차해 피치를 추론**한 것.  
원본(스템·전사)에서 **직접 읽지 않으면** 왜곡이 남는다 — v1 교차검증은 템포 격자용이지,  
피치를 “만들어 내는” 면허가 아님.

산출: `out/20260812_event_pitch_E12_dir_506_indep_octave_t60_90/`  
(exact 27 · octave_fix 10 · mismatch→pyin 68 · cqt_only 33 — 정량 무관, 청취 no-go)

### 다음 방향 메모 (2026-08-12)

rescue = AMT/fuse가 비운 칸 → 참조 피치를 공급하려면 **Transkun 포크 / 다른 AMT / 자작** 중 하나.  
목적상 장기적으로는 **자작**(온셋조건 피치·원본 데이터 기반)에 기울지만, **당장은 다른 AMT 탐색 우선**.  
자작 착수·GT 수집·포크는 이 단계에서 하지 않음.

## AMT 탐색 — Basic Pitch (감사 B) · 청취 대기

Transkun과 **다른 가족**. Dir `piano.wav` 원본 전사만 (추정기 없음).  
산출: `src/exp/s5_midi/clean_amt/out/20260812_clean_amt_basic_pitch_dir_piano_506/`

| window | 506 | BP notes | hit | miss | BP hit ∧ ¬fuse | BP hit ∧ ¬TK | BP hit ∧ ¬fuse∧¬TK |
|--------|----:|---------:|----:|-----:|---------------:|-------------:|-------------------:|
| 30–60s | 127 | 358 | 105 | 22 | 0 | 17 | **0** |
| 60–90s | 138 | 362 | 123 | 15 | 123 | 123 | **123** |

※ 60–90: fuse·clip-Transkun 기준선이 이 구간에 **없음** → “신규 hit”는 구조적으로 전부. 청취로만 판단.  
※ 30–60: BP가 fuse∪TK가 비운 506을 **추가로 메운 건 0**.

```powershell
Set-Location src\exp\s5_midi\clean_amt\scripts
& $py audit_basic_pitch_vs_506.py --config ..\configs\audit_basic_pitch_dir_piano_506.yaml --repo-root <repo>
```

| 파일 | 역할 |
|------|------|
| **`t60_90/piano_listen_t0.mid`** | raw BP · 재생용 |
| `t60_90/piano_506_snap_hits_only_listen_t0.mid` | 506 히트만 스냅 |
| **`t60_90/tk_plus_bp_listen_t0.mid`** | Transkun@60–90 + BP (fuse는 이 구간 없음) |
| `t60_90/transkun_listen_t0.mid` | TK만 |
| `t60_90/bp_added_only_listen_t0.mid` | BP 신규분만 |
| `t30_60/piano_listen_t0.mid` | BP 단독 대조 (506 신규 0 — 합치기 산출 없음) |

배음 판별: TK → TK+BP → bp_added_only (60–90).  
※ velocity 버그(amplitude→1)는 재생성으로 수정됨.


## E10 — 사건 개수 (참고)

| 대상 | 개수 |
|------|------|
| **506 피크 앵커** | **고정** (파일럿 127) — 바뀌지 않음 |
| MIDI **1:1** (피크당 최대 1음, onset→506 스냅) | `n_matched` ≤ 127 · miss = fuse±tol 없음 |
| MIDI **poly** (tol 안 fuse 전부) | 127을 **넘을 수 있음** (어택당 복수 피치) |

이번 런: 1:1 **126/127** (miss 1) · poly **365**.

산출: `out/20260812_event_pitch_E10_dir_506_fuse_mask_t30_60/`

| 파일 | 역할 |
|------|------|
| **`piano_from_506.mid`** | primary = 1:1 × `soft506` |
| `eval_1to1_gated_stem_env.mid` | 게이트엔벨 선택 |
| `eval_1to1_sf_kenv_x_gated.mid` | 곱마스크 선택 |
| `eval_poly_soft506.mid` | poly (밀도↑) |
| `misses_1to1_*.json` | 누락 506 시각 |

```powershell
Set-Location src\exp\s5_midi\event_pitch\scripts
& $py e10_fuse_mask_eval.py --config ..\configs\e10_dir_fuse_mask_eval.yaml --repo-root <repo>
```

청취: primary(1:1) vs 원 piano — **506 클릭 자리에 fuse 피치가 선율로 붙는지**. poly는 밀도 대조.


AMT를 게이트 오디오에 돌린 결과는 기존 전사를 잘라 낸 것과 유사 → **유의미하지 못함**으로 기록.

## E9 — 베일 성운식: **풀 스템 × 마스크 가중** (청취 대기)

원본(풀 piano)에서 피치/배음 유지. 마스크는 성운 detail/다크니스처럼 **가중·선택만**.

| 마스크 | 역할 | 비고 |
|--------|------|------|
| `gated_stem_env` | soft-506 게이트 피아노의 엔벨로프 | 피치 보존 국역 오디오 → weight (당신 아이디어) |
| `sf_kenv` | k_env 소재 SuperFlux | 506 검출 가족 — 게이트보다 **어택 마스크로 적합** 후보 |
| `sf_kenv_x_gated` | 둘의 곱 | 어택 ∩ 피치국소 (primary) |
| `soft506` / `sf_piano` | 커널 / 피아노 SF | 대조 |

적용: (1) 풀 스템 Transkun → 506±50ms에서 `vel×mask`로 픽 (+rescue)  
(2) ByteDance onset×mask → 506마다 argmax

산출: `out/20260812_event_pitch_E9_dir_506_mask_weight_t30_60/`

| 파일 | n | 용도 |
|------|--:|------|
| **`piano_from_506.mid`** | 208 | primary = amt×`sf_kenv_x_gated` |
| `piano_from_506_amt_gated_stem_env.mid` | 140 | 게이트엔벨만 |
| `piano_from_506_onset_sf_kenv_x_gated.mid` | 127 | onset×곱마스크 (1피크1음) |
| `amt_full.mid` | — | 마스크 없는 풀 AMT 대조 |
| `masks/*_pilot.wav` | — | 마스크 소니파이 |

```powershell
Set-Location src\exp\s5_midi\event_pitch\scripts
& $py e9_mask_weight_amt.py --config ..\configs\e9_dir_mask_weight_pilot.yaml --repo-root <repo>
```

청취: primary mid vs 원 piano. 필요 시 gated_stem_env / onset×곱 / amt_full 대조.




| 단계 | 방법 | same±50ms | 청취 |
|------|------|----------:|------|
| E1 | CQT salience | 20% | **no-go** |
| E2 | BP `note` frame | 18% | **no-go** |
| E3 | CQT pitch-SF | 23% | **no-go** |
| E4 | BP `onset` | 15% | **no-go** |
| E5 | Böck ¼음 SF 2D | 14% | **no-go** |
| E6 | 88키 SF | 6% | **no-go** |
| E7 | ByteDance onset | 56% (정량 최선) | **no-go** (음율 실패) |

## 수가 남는가? (판단)

| 방향 | 상태 |
|------|------|
| 피크 리스트에 묻힌 음정 리버스 | **불가** |
| \(t_i\) pitch-wise SF / onset argmax (E3–E7) | **시도 · 음율 실패** |
| 같은 계열 미세 튜닝 | **비권장** |
| **가설 유지 시 좁은 문** | 아래 |
| 506 폐기하고 fuse/clean만 | 별 축 (이미 go 있음) |

### 헤매는 지점 (정확한 병목)

청감은 이미 “506 ≈ 선율 **사건 시각**”을 지지한다. 막힌 것은:

> 폴리포니 스템에서, **알려진 어택 시각**에 대응하는 **선율 음정**을 고르는 일.

E7이 clip AMT와 56% 같아도 음율 no-go인 이유 후보: 참조(밀집 AMT)≠청감 선율, 또는 argmax가 어택 광대역/다른 성부를 집음.

### 가설을 버리지 않을 때의 좁은 문 (미착수)

1. **선율 선택 규칙** — onset 벡터 top-1이 아니라 skyline / 중고역 밴드 / 최고 salience 성부  
2. **읽기 소재 분리** — 시각=506(어택), 피치=harmonic·clip 쪽 스펙트럼만 (perc/tilt가 아님)  
3. **교차가 아니라 게이트** — fuse/AMT 노트를 506에 이식하지 말고, ±506±tol 안의 AMT만 남긴 mid** (506-only는 공백/플래그)  
4. **소수 구간 수동 GT** — 클릭과 같이 들린 음정 10–20개만 적어 E7 등과 대조 (평가 축 교정)

1–3은 새 대가설이 아니라 **같은 병목의 다른 읽기**. 실패하면 그때 506→피치 결합을 접는다.
