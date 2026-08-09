# s4 piano stem event sculpt

BS-Roformer piano stem을 **전처리 → 전장 WAV 청취**로 이산 사건 잔여
후보를 만드는 실험 작업공간이다.

기존 `_onset_*.py` / `stem_validation` / `transcription`과 분리한다.
395 대비 A/B 소니파이·L/R stem 진단 포맷은 만들지 않는다.
산출 WAV 자체가 청취물이다.

## 목표

- 귀납·저누락: 검출기 파라미터 고르기보다 성분 분리 잔여를 듣는다.
- 1차: HPSS / LPC whitening / sinusoidal residual 세 가정을 **병행 산출**.
- 출력은 전장(~117s) stereo FLOAT.

## 입력 (고정)

- `out/stems/Dir/bs_roformer/piano.wav`만 사용
- ODF / 전사 / 395로 마스크하지 않음

## 고정 파라미터 (D-21)

| 패스 | 사건 후보 | 보완 | 값 |
|------|-----------|------|-----|
| HPSS | `hpss_percussive.wav` | `hpss_harmonic.wav` | kernel=31, power=2, margin=1, n_fft=2048, hop=256 |
| LPC | `lpc_residual.wav` | `lpc_synthesis.wav` | order=24, frame=2048, hop=512, pre-emphasis=0.97 |
| Sinusoidal | `sine_residual.wav` | `sine_tonal.wav` | local-max ∧ mag≥frame p90, n_fft=2048, hop=256 |

청취 파일만 `peak>0.98`이면 파일 단위 soft limit. 패스 내부 이득 조절 없음.

## 출력

```
out/stems/Dir/event_sculpt/
  hpss_percussive.wav
  hpss_harmonic.wav
  lpc_residual.wav
  lpc_synthesis.wav
  sine_residual.wav
  sine_tonal.wav
  sculpt_manifest.json
  sculpt_determinism.json   # --determinism-check 시
```

## 실행

프로젝트 루트 PowerShell, 공유 venv:

```powershell
$py = "E:\game\Music Hermeneutic AI\.venv\Scripts\python.exe"
& $py "src\exp\s4_piano\stem_event_sculpt\run_passes.py"
& $py "src\exp\s4_piano\stem_event_sculpt\run_passes.py" --determinism-check
```

출력을 듣고 파라미터를 바꾸지 않는다.

## 청취 순서 (권장)

1. `hpss_percussive` → `hpss_harmonic`
2. `lpc_residual` → `lpc_synthesis`
3. `sine_residual` → `sine_tonal`

각 쌍에서 “사건이 남는지 / 무엇이 빠졌는지”만 기록. 395와 비교는 작업 산출이 아님.

## 청취 판정 (세션 13)

| 산출 | 판정 |
|------|------|
| **hpss_percussive** | **주 후보** — 건반 타건 순간 포괄 |
| **lpc_residual** | **보조(조건부)** — 이산 성공, 볼륨 편차·누락 → 값 조정 테스트 시 |
| **lpc_synthesis** | **유효** — 사건↔링잉 대비가 raw보다 돋봄 |
| hpss_harmonic | 전자피아노; sine_residual보다 raw 가까움 → 가벼운 전처리 후보 |
| sine_residual | 울림·링잉 큼; 관심에서 다소 멀음; 중저역 미결 |
| sine_tonal | 배음 포락선; 본 프로젝트 사건 축은 의문 |

출력을 듣고 1차 고정 파라미터를 바꾸지 않는다. LPC 값 조정은 별도 D-21 선언 후.

## Workspace

`runtime/`, `work/`는 이 디렉터리 아래 격리·gitignore.
