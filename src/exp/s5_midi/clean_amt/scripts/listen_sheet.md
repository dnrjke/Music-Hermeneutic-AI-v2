# clean_amt 청취 루브릭 시트

계획: Docs/piano_stem_to_midi_plan_cursor_grok_4.5.md §6.3  
성공 문구: 스케치 MIDI / 편집 시작점

## 정책

- **기본 GT** = MAPS AkPnBcht `polyphony_baroque`
- MAESTRO = 선택 교차 (go)

---

## GT / dry (요약)

| 런 | 판정 |
|----|------|
| MAPS GT | go — 원곡을 연주했다고 할 수 있음 (note length만 GT보다 김/분절) |
| MAESTRO | go — GT와 사실상 동일 |
| dry_bach | soft go — 봐줄 만함·쓸 만함 |

---

## Dir 다층 입력 → MIDI (t=30–60s · Transkun) — **역할 잠금 2026-08-12**

단일 mid 즉시 납품은 아님. 청취 후 **레이어 역할**을 임시 확정.

### 역할 표 (사용자 판정)

| 우선 | 입력 → MIDI run | notes | 역할 |
|-----:|-----------------|------:|------|
| 1 | **dir_clip** (`stem_dir_clip`, raw BS piano) | 878 | **본선 bed** — 무난·안정 · 배음이 이 중 가장 음악적 · **원곡에 제일 가깝게 들림** |
| 2 | **hpss_harmonic** | 462 | **clip rescue (임시)** — 주방식으로 쓸 수도 있으나 일단 rescue. clip이 놓친 사건·배음이 이쪽에 더 잘 보인 경우 있음 (우위 미검증) |
| 3 | **lpc_synthesis** | 448 | **구조/신뢰 축** — note length는 약함 · 누락·배음 적음 · **노트 배치가 더 정갈** · 신뢰성은 clip/harmonic보다 높아 보임. (당장 듣기엔 clip이 원곡감) |
| 4 | **hpss_percussive** | 194 | **왼손·저역 보강** — 고역은 거의 없음 · 저역 사건은 더 잘 잡힘 |
| 5? | **lpc_residual** | 476 | **onset 시점만?** — 길이 극단적으로 짧고 생략 많음 · 부가 사용 여부 미결 |
| — | sine_tonal / sine_residual | 416 / 450 | **제외** — 변형이 눈에 띔 |
| — | tilt_high / k_env material | 166 / 158 | **제외** — 저역·희박, 뚜렷이 좋지 않음 |

### 한 줄 스택 (임시)

```
본선: dir_clip
  └ rescue: hpss_harmonic
구조 앵커: lpc_synthesis
저역/왼손: hpss_percussive
onset 힌트?: lpc_residual (옵션)
제외: sine_* · tilt_*
```

### 개별 메모 (원문 요약)

- **dir_clip**: 무난·안정 · 배음 음악적 · 이 중 원곡 최근접.
- **hpss_harmonic**: 주방식 후보 가능하나 당분간 clip rescue.
- **lpc_synthesis**: 구조에 의지하고 싶음 · 길이는 약 · 정갈·신뢰↑.
- **hpss_percussive**: 왼손/저역 탐지·보강.
- **lpc_residual**: 시점용 가능, 길이·생략 때문에 부가 여부만 남음.
- **sine 계열**: 고려 대상 아님.

---

## 닫힘 / 재개 조건

- M1 단일 스템 MIDI **즉시 사용**은 여전히 no (잡·누락·파편 — 초기의 dir_clip 단독 판정).
- 다층 역할표 확보 후 **스템 정상화(v1)** 착수 → 아래 런 G.

---

## 런 G — stem_norm v1 → MIDI (청취 대기)

레시피: `norm_v1_attack_lowblend` — piano sustain duck(0.4) + perc ≤200 Hz blend(0.3)  
코드: [`../../stem_norm/`](../../stem_norm/)  
WAV: `src/exp/s5_midi/stem_norm/out/normalized_v1.wav`  
AMT: `20260812_clean_amt_transkun_stem_dir_norm_v1` · **notes=451** (clip=878)

| 비교 | bed |
|------|-----|
| baseline | `stem_dir_clip` mid + raw piano |
| 후보 | norm_v1 wav + 위 mid |

**판정 기준**: 건반 누락↓ **그리고** 울림/유령↓ (한쪽만이면 no-go)

| # | 항목 | 점수/메모 |
|---|------|-----------|
| 1 | 건반 누락 (clip 대비) | 일부는 더 잡힘 · **전체 no** |
| 2 | 울림/유령 전사 | (부분 이득 가능하나) 상쇄됨 |
| 3 | 저역/왼손 | 저역 쪽 편향 (의도했으나 고역 희생) |
| 4 | 원곡감·변형 | **피치가 원곡 대비 틀어짐** · 고역 사건↓ |

**go / no-go (norm_v1)**: **no-go**

### 정량 (착각 여부 — 고역)

동일 구간 Transkun mid 비교 (clip=미리 자른 30s 파일 · norm=풀파일 30–60s):

| | n | mean pitch | ≥72 (고역) | ≥84 |
|--|--:|----------:|----------:|----:|
| dir_clip | 878 | 65.3 | **345** | **102** |
| norm_v1 | 451 | 63.5 | **165** | **25** |

→ 고역 감소는 **착각 아님**. vhigh(≥84)는 약 1/4.

onset ±50ms로 clip에 붙인 norm 노트: 448/451 · 같은 pitch 390 · ±12옥타브 8.  
매칭된 쪽은 clip과 대체로 같으나, **고역 노트가 통째로 사라지며** 저역 blend·duck이 스펙트럴 무게를 아래로 밀어 AMT/청감 모두 “피치 틀림·고역 못 잡음”으로 읽히기 쉽다.

- go → 전곡/파라미터 미세조정 또는 융합 규칙 검토  
- no-go → **v2는 harmonic 소량이 아니라**, 저역-only blend·강한 duck 재검토 또는 **MIDI 융합(본선 clip + harmonic rescue)** 쪽이 맞음

## 자유 메모

- 우위 clip vs harmonic은 미검증 — harmonic은 임시 rescue.
- norm_v1 note count 451 ≠ 성공 지표 (clip 878은 잡 포함).
- norm_v1 **폐기 후보** (청취 no-go + 고역 정량 악화).

---

## 런 H — midi_fuse v1 (청취 대기)

오디오 정상화 대신 **MIDI 스택 융합**. 창 abs **30–60s** (clip 로컬 0–30).  
규칙: clip 전부 유지 + rescue는 동일 pitch가 ±30ms 안에 없을 때만 추가.

| name | 산출 | base | +rescue | fused |
|------|------|-----:|--------:|------:|
| **clip⊕harmonic** | `midi_fuse/out/20260812_midi_fuse_clip_harmonic/` | 454 | **+37** | 491 |
| **clip⊕synthesis** | `midi_fuse/out/20260812_midi_fuse_clip_synthesis/` | 454 | **+12** | 466 |

청취 bed: raw piano (또는 dir_clip wav) vs 각 `piano.mid`.  
비교 기준: clip-only mid 대비 rescue가 **놓친 건반**을 메우는지, 잡만 늘리는지.

| 버전 | go/no-go | 메모 |
|------|----------|------|
| clip⊕harmonic | **go · 잠금** | Transkun **짧은 노트** → harmonic으로 **길이·밀도 보완(rescue)**. 실제 사건 증가=**미검토**. **당장 근거=감상** (“듣기 좋다”=원곡 해치지 않고 더 재현). |
| clip⊕synthesis | 보류 | rescue 12음 low-piano 클릭으로 별도 판정 (본선 교체 아님) |

### rescue-only 소니파이 (low piano ×0.20 · 3kHz 클릭)

기존 Dir low-piano 클릭 방식. **추가된 rescue onset만** 찍음.

| 버전 | n | 전곡 | 창 t=30–60 |
|------|--:|------|------------|
| clip⊕harmonic rescue | **37** | `midi_fuse/out/20260812_midi_fuse_clip_harmonic/clip_harmonic_rescueOnly3k_low_g0p20_클릭_p37.wav` | `…_t30_60_클릭_p37.wav` |
| clip⊕synthesis rescue | **12** | `midi_fuse/out/20260812_midi_fuse_clip_synthesis/clip_synthesis_rescueOnly3k_low_g0p20_클릭_p12.wav` | `…_t30_60_클릭_p12.wav` |

스크립트: `midi_fuse/scripts/sonify_rescue_only.py`

## 자유 메모

- 우위 clip vs harmonic — **clip⊕harmonic go 잠금** (감상 우선; 사건 증분은 미검토; Transkun 짧은 노트 보완 목적 포함).
- Basic Pitch: 왼손/배음 등으로 **풍성함** 인상(참고; 본선 go 아님).
-
