# clean_amt — 클린 슬레이트 AMT 실행 노트

상위: [`../README.md`](../README.md) · 계획: [`Docs/piano_stem_to_midi_plan_cursor_grok_4.5.md`](../../../../Docs/piano_stem_to_midi_plan_cursor_grok_4.5.md)

**목표 문구**: 스케치 MIDI / 편집 시작점 (상용급 전사 아님).  
**금지**: `s4_piano` import · s4 venv · `out/transcription/` 덮어쓰기.

---

## 상태

| 단계 | 상태 |
|------|------|
| K0 골격 | 완료 |
| M0 선정·입력 표 | 완료 — 파일럿 Transkun 2.0.1 / 감사 Basic Pitch 0.4.0 (**506 대조 런 완료 · 청취 대기**) |
| M1 파일럿 | **닫음+역할표** — GT/dry go · Dir 다층 역할 · **stem_norm v1 전사 완료 · 청취 대기** (451n vs clip 878) |
| M2–M4 | **보류** — 다음이면 다층 융합 규칙 또는 스템 정상화 |

### 닫힘 한 줄

dry/GT는 Transkun 스케치 가능. Dir는 raw clip만으로 납품 no이나, 청취로 **다층 역할**이 잡힘(본선 clip · rescue harmonic · 구조 synthesis · 저역 perc). sine/tilt 제외. M2/`via_764` 보류.

청취 상세: [`scripts/listen_sheet.md`](scripts/listen_sheet.md)

---

## M0 체크리스트 (클린 플랜 §8)

- [x] 후보 모델 ≤2 확정 (아래 shortlist)
- [x] Python/CUDA/패키지 pin → [`env/requirements.txt`](env/requirements.txt) (`env/.venv`, Py3.11, torch cu121)
- [x] 입출력·명명 규칙 (이 문서 + configs)
- [x] manifest 스키마 (아래)
- [x] 파일럿 입력 표 (아래)
- [x] 성공/실패 한 줄 (아래)

### 성공 / 실패 (한 줄)

- **성공**: GT 파일럿에서 mid가 열리고, 청취·또는 note F1로 “편집 시작점”으로 쓸 수 있다.
- **실패**: 유령 음·옥타브 오류가 심해 처음부터 치는 편이 낫다 → 모델 교체 또는 입력(스템) 재검토.

---

## 파일럿 입력 표 (읽기 전용)

경로는 리포 루트 기준. 복사하지 말고 config에만 적는다.

| 역할 | 경로 | GT |
|------|------|-----|
| **GT 1차 (기본)** | `audio/midi_eval/paired_gt/maps_akpnbcht/polyphony_baroque/audio.wav` | `.../gt.mid` |
| **GT 교차 (선택)** | `audio/midi_eval/paired_gt/maestro_v3/polyphony_baroque/audio.flac` | `.../gt.midi` |
| **dry** | `audio/midi_eval/dry_solo/bach_wtc_martins_lp_t120_60s.wav` | 없음 |
| **본선 스템** | `audio/midi_eval/target_stems/Dir/bs_roformer_piano_t30_60s.wav` | 없음 |
| 풀 스템 (포인터) | `out/stems/Dir/bs_roformer/piano.wav` | 없음 |

config: [`configs/pilot_gt.yaml`](configs/pilot_gt.yaml) · [`configs/pilot_stem.yaml`](configs/pilot_stem.yaml) · [`configs/pilot_dry.yaml`](configs/pilot_dry.yaml)

---

## AMT 모델 shortlist (≤2 — **선정 완료**)

s4 transcription 결과/venv를 **재사용하지 않음**. 독립 재선정·독립 `env/.venv`.

| # | 후보 | 레포 / 가중치 | 라이선스 | 비고 | 선택 |
|---|------|---------------|----------|------|------|
| A | **Transkun 2.0.1** | [Yujia-Yan/Transkun](https://github.com/Yujia-Yan/Transkun) · PyPI `transkun` · 기본 shipped ckpt (V2, no pedal ext, aug) | **MIT** | 피아노 전용; MAESTRO SOTA급 계열. **파일럿** | [x] 파일럿 |
| B | **Basic Pitch 0.4.0** | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) · ICASSP 2022 기본 가중치 | **Apache-2.0** | 경량·범용 폴리포니. **감사/교차** | [x] **506 대조 런** |

**506 감사 (2026-08-12)**: [`out/20260812_clean_amt_basic_pitch_dir_piano_506/`](out/20260812_clean_amt_basic_pitch_dir_piano_506/) · [`vs_506.md`](out/20260812_clean_amt_basic_pitch_dir_piano_506/vs_506.md)  
30–60: hit 105/127 · fuse∪TK 대비 **신규 0**. 60–90: hit 123/138 (기준선 없음 → 청취 본창).  
스크립트: `scripts/audit_basic_pitch_vs_506.py` · config: `configs/audit_basic_pitch_dir_piano_506.yaml`

**선정 이유**: A는 피아노 AMT에 맞고 pip·라이선스가 깨끗함. B는 의존성·런타임이 달라 같은 입력에서 독립 감사에 적합.  
**품질 검토 시점**: 사용자 청취 go/no-go (60–90 raw / 506 snap).

파일럿은 **A만** 켠다. `model_id: transkun` / `model_version: 2.0.1`. 감사 B는 위 스크립트.

---

## Manifest 스키마 (`out/<run_id>/manifest.json`)

```json
{
  "run_id": "string",
  "created_utc": "ISO-8601",
  "track": "clean_amt",
  "model_id": "string",
  "model_version": "string",
  "config_path": "string",
  "input": {
    "path": "string",
    "sha256": "string",
    "start_s": 0.0,
    "end_s": null,
    "role": "gt_maestro_polyphony_baroque | gt_maps_polyphony_baroque | dry_bach | stem_dir_clip"
  },
  "preprocess": { "mono": true, "normalize": "peak|rms|none", "hp_hz": null },
  "postprocess": { "min_dur_s": 0.0, "min_vel": 0, "merge_ms": null, "soft_quantize": false },
  "outputs": {
    "piano_mid": "piano.mid",
    "notes_json": "notes.json",
    "preview_wav": null,
    "metrics_json": null
  },
  "notes": "string"
}
```

`notes.json`: `[{ "onset_s", "offset_s", "pitch", "velocity" }, ...]`

`run_id` 예: `20260812_clean_amt_<model>_gt_maestro_polyphony_baroque`

---

## 산출물

`out/<run_id>/` → `piano.mid` · `notes.json` · `manifest.json` · (선택) `preview.wav` · `metrics.json`

---

## 실행

```powershell
# 리포 루트에서
$py = "src\exp\s5_midi\clean_amt\env\.venv\Scripts\python.exe"
& $py src\exp\s5_midi\clean_amt\scripts\transcribe.py --config src\exp\s5_midi\clean_amt\configs\pilot_gt.yaml
& $py src\exp\s5_midi\clean_amt\scripts\evaluate_gt.py --run-dir src\exp\s5_midi\clean_amt\out\<run_id>
```

파일럿: Transkun (`model_id: transkun`). `evaluate_gt.py`는 GT mid와 `notes.json` 비교.

---

## M1+ 청취

[`scripts/listen_sheet.md`](scripts/listen_sheet.md)
