# stem_norm — Dir 피아노 스템 정상화 (AMT 전단)

상위: [`../README.md`](../README.md) · 형제: [`../clean_amt/`](../clean_amt/)

**목표**: 단일 WAV를 만들어 Transkun이 **건반 누락↓ · 울림/유령↓** 하도록 한다 (노트 수↑ 아님).  
**금지**: `s4_piano` import · s4 venv · MIDI 스택 융합(별 트랙).

## 상태

| 단계 | 상태 |
|------|------|
| v1 `norm_v1_attack_lowblend` | **no-go** — 피치 틀림 · 고역↓ (정량: ≥84 notes 102→25) |
| v2 (harmonic blend) | v1 no-go 시에만 |

## 레시피 v1

1. Bed: `out/stems/Dir/bs_roformer/piano.wav`
2. Sustain duck: 느린 RMS 기반 soft mask를 **약하게** 적용 (어택≈1, sustain&lt;1)
3. Lowband blend: `hpss_percussive` ≤200 Hz 소량 가산
4. Peak normalize 0.95 · harmonic 전대역 믹스 **없음**

## 입출력

| | 경로 |
|--|------|
| in (RO) | `out/stems/Dir/bs_roformer/piano.wav` |
| in (RO) | `out/stems/Dir/event_sculpt/hpss_percussive.wav` |
| out | `out/<run_id>/normalized.wav` · `manifest.json` |
| AMT | [`../clean_amt/configs/pilot_norm_v1.yaml`](../clean_amt/configs/pilot_norm_v1.yaml) |

## 실행

```powershell
$py = "src\exp\s5_midi\clean_amt\env\.venv\Scripts\python.exe"
& $py src\exp\s5_midi\stem_norm\scripts\normalize_v1.py
& $py src\exp\s5_midi\clean_amt\scripts\transcribe.py --config src\exp\s5_midi\clean_amt\configs\pilot_norm_v1.yaml
```

청취: [`../clean_amt/scripts/listen_sheet.md`](../clean_amt/scripts/listen_sheet.md) 런 G.
