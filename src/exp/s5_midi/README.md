# s5_midi — 피아노 → MIDI 실험 (s4와 분리)

- **형제**: `s1_proto` · `s2_1d` · `s3_2d` · `s4_piano` · **`s5_midi`**
- **트랙**: `clean_amt/` · `midi_fuse/` · `stem_norm/` · `via_764/` · **`event_pitch/`**
- **비의존**: `s4_piano` import 금지. 형제 트랙끼리도 **import·out 공유 금지** (WAV/JSON 읽기 전용만).

## 트랙

| 경로 | 상태 | 계획서 |
|------|------|--------|
| [`clean_amt/`](clean_amt/) | **M1 닫음** — GT/dry go · Dir 다층 역할표 · 단일 mid 납품 no | [`Docs/piano_stem_to_midi_plan_cursor_grok_4.5.md`](../../../Docs/piano_stem_to_midi_plan_cursor_grok_4.5.md) |
| [`stem_norm/`](stem_norm/) | **v1 no-go** — duck+저역 blend → 피치·고역 악화 | listen_sheet 런 G |
| [`midi_fuse/`](midi_fuse/) | **go (Dir 선율·감상용 MIDI)** — clip⊕harmonic + 풀길이. harmonic은 **듣기 좋아서** 채택(사건 추가 아님). 링잉 연구 시 piano-only. [`MIDI_GUIDE.md`](midi_fuse/MIDI_GUIDE.md) | listen_sheet 런 H |
| [`event_pitch/`](event_pitch/) | E1–E12 추정기 **no-go** · 후속 **비급** (MIDI 보강·764↔음 매칭 의의) | [`Docs/piano_event_pitch_plan_cursor_grok_4.5.md`](../../../Docs/piano_event_pitch_plan_cursor_grok_4.5.md) |
| [`via_764/`](via_764/) | D1 피치 **no-go** · **비급** — 764는 사건 도구(목적≠MIDI 본선) | [`Docs/piano_midi_via_dir764_plan_cursor_grok_4.5.md`](../../../Docs/piano_midi_via_dir764_plan_cursor_grok_4.5.md) |
## 입력 (읽기 전용)

- [`audio/midi_eval/`](../../../audio/midi_eval/) — GT 페어·dry·Dir stem 클립 ([SOURCES_DIGEST](../../../audio/midi_eval/SOURCES_DIGEST.md))
- 풀 스템 포인터: `out/stems/Dir/bs_roformer/piano.wav` (복사하지 않음)

## 성공 문구

스케치 MIDI / 편집 시작점. 상용급 자동 전사·악보 조판은 비범위.
