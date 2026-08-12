# midi_fuse — Dir 다층 Transkun MIDI 스택 융합

상위: [`../README.md`](../README.md) · 역할표: [`../clean_amt/scripts/listen_sheet.md`](../clean_amt/scripts/listen_sheet.md)  
**MIDI 소개 (청취)**: [`MIDI_GUIDE.md`](MIDI_GUIDE.md) ← mid 파일별 설명

**목표**: 본선 `dir_clip` mid를 유지한 채 rescue 층만 얹어 스케치 밀도↑.  
**금지**: s4 import · 스템 오디오 재믹스(stem_norm과 별개).

## 시간축

| 소스 | 파일 시간 | abs (원곡) |
|------|-----------|------------|
| `stem_dir_clip` | 로컬 0 = piano **30s** (60s짜리 클립) | onset_local + 30 |
| `stem_dir_hpss_harmonic` / `lpc_synthesis` | config start=30 | 이미 abs |

융합 창(파일럿): **abs 30–60s** (= clip 로컬 0–30).  
**풀 길이** (2026-08-12): `fuse_v1_full.py` → `piano_harmonic_full` (아래).

## 규칙 v1 (고정)

**본선**: clip 노트 전부 유지 (창 안).

**Rescue 추가** (harmonic 또는 synthesis):  
같은 pitch 가 본선에 `|onset_diff| ≤ 30ms` 이면 스킵.  
아니면 rescue 노트를 추가 (`source` 태그).

- `fuse_clip_harmonic` — rescue = hpss_harmonic · **창 30–60만** (**go**)  
  - Transkun의 **짧은 노트**를 길이·밀도로 보완하는 rescue이기도 함  
  - **실제 사건 증가 여부는 미검토** · 당장 근거는 **감상**  
    (“듣기 좋다” = **원곡을 해치지 않고 더 재현**한다고 해석)  
- `fuse_clip_synthesis` — rescue = lpc_synthesis · 30–60  
- **`fuse_piano_harmonic_full`** — 같은 규칙 · **풀 스템**

**초점**: 감상용 MIDI 전사. 링잉·사건 연구는 `piano_base_only` (harmonic 없음).  
**BP**: 왼손/배음 등으로 풍성해 보임 — 참고 인상(본선 go 아님).
duration/onset 스냅·삭제(고스트 제거)는 v1 비범위.

## 실행

```powershell
$py = "src\exp\s5_midi\clean_amt\env\.venv\Scripts\python.exe"
& $py src\exp\s5_midi\midi_fuse\scripts\fuse_v1.py
# 풀 길이 (선행: clean_amt piano_full + hpss_harmonic_full Transkun)
& $py src\exp\s5_midi\midi_fuse\scripts\fuse_v1_full.py
```

산출: `out/<run_id>/` → `piano.mid` · `notes.json` · `manifest.json`

| 런 | 구간 | n |
|----|------|--:|
| `20260812_midi_fuse_clip_harmonic` | 30–60 | 491 (go) |
| `20260812_midi_fuse_piano_harmonic_full` | **0–끝** | 1753 (base 1667 +harm 86) |

## rescue-only 소니파이

```powershell
& $py src\exp\s5_midi\midi_fuse\scripts\sonify_rescue_only.py
```

low piano ×0.20 + 3kHz 클릭 · **추가 onset만**. 전곡 + `t30_60` 크롭.
