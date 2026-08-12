# 피아노→MIDI용 소스 자산 — 버전업 (v1)

- **작성**: Cursor Grok 4.5
- **일자**: 2026-08-12
- **성격**: 기존 MIDI 계획서의 **입력·평가 소스** 보완. 알고리즘 본문은 바꾸지 않음
- **부모 문서** (본문 유지, 여기로 포인터만):
  - [`piano_stem_to_midi_plan_cursor_grok_4.5.md`](piano_stem_to_midi_plan_cursor_grok_4.5.md) — 클린 슬레이트 AMT
  - [`piano_midi_via_dir764_plan_cursor_grok_4.5.md`](piano_midi_via_dir764_plan_cursor_grok_4.5.md) — 506/764 onset 골격
- **조달·품질 규율 참조 (v1, 읽기 전용)**:  
  `E:\game\Music Hermeneutic AI\audio\control\README.md`  
  (D1 음성 대조 조달은 **닫힘** — 이 문서는 **피아노→MIDI** 목적의 재사용·추가 조달이며 D1 세트를 재개하지 않음)

---

## 0. 이 문서가 답하는 것

1. **이미 있는** 피아노(또는 피아노에 가까운) 소스를 MIDI 시도에 어떻게 쓸지  
2. **더 구해오면 좋은** 소스와 우선순위  
3. control README에서 **그대로 가져올 규율** vs **MIDI용으로 바꾸어야 할 기준**

---

## 1. 이미 있는 소스 — 역할 재배치

control README와 v2 Dir 작업에서 확보·사용된 것을 **MIDI 파이프라인 역할**로만 다시 붙인다. D1 hub 배율·채택 여부와는 무관하다.

### 1.1 솔로 피아노 (깨끗함 · 방법 천장)

| 자산 | 위치(v1) | MIDI 시도에서의 역할 |
|------|----------|----------------------|
| **Bach WTC – Martins (LP side1)** | `audio/control/Bach WTC - Martins (LP side1).flac` | **방법 검증용 dry 솔로**. 분리 bleed 없이 onset·피치·duration 파이프라인이 “깨끗한 피아노에서 도는지” 본다. 6분+·무손실(README 실측 19 kHz 대역, energy_ok). |
| 원본 LP 면 | `audio/control/_source/` (해당 disc side) | 재전처리·구간 재절단 필요 시만. **곡 단위로 잘게 자르지 말라**는 README 규율은 D1용; MIDI 파일럿은 **짧은 구간 추출 허용**(별 파일로 복사). |

**쓰지 않는 이유까지 명시된 것**

| 자산 | README 판정 | MIDI에서의 취급 |
|------|-------------|-----------------|
| Glassworks Opening | **손실** (≈9 kHz 브릭월) → D1 제외 | **GT/방법 검증에 쓰지 않음**. 상단 밴드 죽음이 mel·피치에도 인공물을 만듦. |
| Orphee Suite | 손실 제외 | 동일 — 제외 |
| Mozart KV550 악장 | 오케스트라 대조 (채택) | 피아노 MIDI 본선 **아님**. “스템이 아닌 풀믹스 전사” 스트레스 테스트에만 선택. |

### 1.2 본선 대상 — 분리 피아노 스템 (현실 조건)

| 자산 | 역할 |
|------|------|
| **Dir · BS-Roformer piano stem** | **본선 목표 입력**. 506/764 경로·클린 AMT 경로 모두 여기가 최종 적용 대상. |
| Dir · Demucs 등 타 모델 piano (있는 경우) | bleed/누락 **감사**. 본선 하나로 고정하지 말 것(세션 합의: BS 주 참조). |
| AS 등 타 곡 bs_roformer piano | 일반화 파일럿 2호(선택). Dir 성공 전 필수는 아님. |
| Dir 원곡 (`102 - Dir.wav`) | 스템 vs 풀믹스 전사 차이·소니파이 bed. MIDI GT가 아님. |

### 1.3 기호·악보 쪽 (오디오 아님)

| 자산 | 역할 | 주의 |
|------|------|------|
| MuseScore 피아노 편곡 (Dir 관련, 팬 편곡) | **느슨한 구조 참조**(조성·프레이즈). note-level GT로 쓰지 않음. | 공식 MIDI 아님. 정렬·편곡 차이 큼. |
| 만료된 귀카피 MIDI 링크 | 사용 불가 | 재조달 후보로만 기록 |

### 1.4 재사용 시 권장 순서 (추가 구매 없이)

```
A. Bach WTC 30–90초 구간
   → 클린 AMT 또는 피치모듈이 dry에서 도는지 (천장)

B. 동일 구간을 “인위 믹스”(선택: 약한 드럼/노이즈 얹기) 후 재전사
   → 분리 전 스트레스 (선택)

C. Dir BS piano stem + (506 경로면) 506 onset
   → 현실 조건 본선

D. 청취 A/B: stem bed · MIDI 렌더 · 기존 506 클릭 소니파이
```

A에서 크게 실패하면 C의 실패는 “스템 탓”이 아니라 **방법 탓**으로 읽는다.

---

## 2. control README에서 가져올 규율 (MIDI용 번역)

| README 교훈 | MIDI 소스 조달에의 적용 |
|-------------|-------------------------|
| 확장자≠무손실 · `check_provenance` / 스펙트럼 절벽 | 새 피아노·MIDI 페어도 **무손실 또는 합성 렌더**만. mp3→flac 금지. |
| 천장 점유율로 채택하지 말 것 | MIDI용 선정 기준도 **리미팅 %가 아님**. 페달·폴리포니·어택 선명도·GT 유무. |
| 긴 트랙이 짧은 둘보다 낫다 (D1 타일) | MIDI 파일럿은 **짧은 구간으로 충분**. 전곡은 방법 고정 후. |
| 음성 대조 조달 **닫힘** | D1 세트에 여덟 번째 대조를 넣지 말 것. 새 파일은 `audio/control`에 넣더라도 **역할 태그를 MIDI-eval로 분리**. |
| LP 간섭음·가짜 코덱 절벽 | 아날로그 립을 GT 정렬에 쓸 때 고역 피치/노이즈 오탐 가능 → dry 검증은 Bach로 하되, **정량 F1 GT는 디지털/합성 페어를 선호**. |

---

## 3. 더 구해오면 좋은 것 — 우선순위

D1용 4순위(타악 등)와 **무관**. 아래는 **피아노→MIDI 전용** wishlist.

### P1 ★ 정렬된 피아노 오디오 ↔ MIDI (또는 MusicXML) 페어

**가장 값어치 큼.** 정량 note F1·onset F1을 열 수 있다.

구하기 좋은 형태(아무거나 1세트면 시작):

| 형태 | 예 | 비고 |
|------|-----|------|
| 공개 AMT 벤치 일부 | MAESTRO / GiantMIDI 일부 트랙, MAPS 등 | 라이선스·용량 확인 후 **짧은 클립만** 복사 |
| 상용/자작: MIDI → 고품질 피아노 샘플러 렌더 | 동일 MIDI를 GT로 보관 | **완벽한 시간 정렬**. 방법 천장·회귀 테스트용 최고 |
| 연주 MIDI + 동시 녹음 | Disklavier / 디지털 피아노 USB-MIDI | 현실 마이크로폰 색 + GT |

**최소 스펙 제안**

- 무손실 WAV/FLAC ≥ 44.1 kHz  
- 동반 `.mid` (또는 note JSON)  
- 길이 30초–3분 클립 2개: (1) 단성·느린 화성 (2) 밀집·페달  

Dir 공식 MIDI가 없어도, P1 하나로 **파이프라인 회귀**가 가능해진다.

### P2 솔로 피아노 dry (Bach와 성격이 다른 것) — 선택

Bach WTC는 바로크·평균율·비교적 또렷한 어택. 다음에 하나 있으면 좋은 축:

- **페달이 많은 낭만** (쇼팽 녹턴 일부 등) — note-off·지속음 스트레스  
- **현대/미니멀 반복** — onset 밀도·유령 음  

무손실 CD/LP 립. Glassworks급 **손실 금지**.  
D1에 넣지 말 것; MIDI-eval 폴더로.

### P3 Dir에 가까운 “믹스 속 피아노” + 가능하면 분리 전·후

| 구하면 | 쓰는 법 |
|--------|---------|
| 피아노+리듬 섹션이 있는 **무손실 멀티트랙** 또는 스템 공개곡 | 분리 모델 없이 “진짜 피아노 트랙” vs “분리 스템” 전사 차이 |
| 동일 곡의 **공식/팬 MIDI** (정렬 가능 시) | Dir MuseScore보다 강한 참조 |

없으면 Dir BS stem만으로도 본선은 진행 가능. P3는 **일반화·감사**.

### P4 명시적으로 구하지 않아도 되는 것

- D1용 오케스트라·합창·타악 **추가** (조달 닫힘 + MIDI 무관)  
- 손실 스트리밍 립  
- “MIDI만 있고 오디오 없는” 악보만 (청취 A/B 불가)  
- 라이브 관객 노이즈가 큰 솔로 (onset 오염)

---

## 4. 권장 디렉터리 (제안, 기존 control과 분리)

v1 `audio/control`은 D1 규율·오염 방지(`--all` 등)가 얽혀 있다. MIDI용은 **역할이 다르므로** 복사본·별 트리를 권장한다.

```
# 제안 (v2 쪽 예시)
audio/midi_eval/
  README.md                 # 이 문서 요약 + 라이선스
  dry_solo/
    bach_wtc_clip_*.wav     # control에서 구간 복사
  paired_gt/
    <id>/audio.wav
    <id>/gt.mid
    <id>/manifest.json
  target_stems/
    Dir_bs_roformer_piano.wav   # 또는 심볼릭/경로 포인터만
```

control 원본은 **이동·삭제하지 않음**. 구간 복사 + `source_path`를 manifest에 기록.

---

## 5. 두 계획서에의 매핑

| 소스 | 클린 슬레이트 AMT | 506/764 활용 |
|------|-------------------|--------------|
| Bach dry 클립 | M1 파일럿 입력 후보 | 피치 모듈만 단독 검증(onset은 별도 peak 또는 AMT) |
| P1 오디오↔MIDI 페어 | 정량 성공 기준 보강 | note-off/피치 정책 튜닝의 숫자 앵커 |
| Dir BS piano | 본선 | 본선 + 506 onset |
| MuseScore | 느슨한 청취 가이드 | 동일 — GT 아님 |
| Glassworks 등 손실 | **금지** | **금지** |

---

## 6. 조달 체크리스트 (새 파일을 받을 때)

- [ ] 무손실인가 (스펙트럼 절벽·provenance)  
- [ ] 역할 태그: `dry_solo` / `paired_gt` / `mix_or_stem` / `score_ref_only`  
- [ ] GT가 있으면 정렬 방법 한 줄 (동일 렌더 / 수동 / 벤치 공식)  
- [ ] D1 `audio/control` 채택 절차를 **요구하지 않음** (hub 배율 불필요)  
- [ ] 라이선스·재배포 가능 여부 한 줄  

---

## 7. 적재 현황 (2026-08-12)

실파일은 **`audio/midi_eval/`**. 출처·라이선스 정본: [`../audio/midi_eval/SOURCES_DIGEST.md`](../audio/midi_eval/SOURCES_DIGEST.md).

| 확보 | 상태 |
|------|------|
| Bach dry 클립 | ✅ `dry_solo/` (MIDI GT **비부착**) |
| Dir BS piano 클립 | ✅ `target_stems/Dir/` |
| 합성 페어 | ✅ `paired_gt/synth_additive_*` |
| MAESTRO 실연 페어 | ✅ `paired_gt/maestro_v3/` ×4 (목적 선별, CC BY-NC-SA 4.0) |
| MAPS AkPnBcht MUS | ✅ `paired_gt/maps_akpnbcht/` ×4 (타 패키지는 digest에만) |
| MAESTRO MIDI-only zip | ✅ `upstream_midi_only/` |
| MAPS 타 패키지 / Mutopia+SF | ❌ digest “못 구한 것” |

---

## 8. 버전 이력

| ver | 일자 | 내용 |
|-----|------|------|
| **v1** | 2026-08-12 | 초판. 기존 두 MIDI 계획서와 control README를 포인터로 연결. 재사용(Bach·Dir stem) + wishlist(P1–P3). |
| **v1+적재** | 2026-08-12 | `audio/midi_eval` 적재·`SOURCES_DIGEST.md`. Dir AI MIDI 배제, Bach MIDI GT 비사용 명시. |

후속 개정 시 이 파일·digest만 올리고, 부모 계획서는 **상단 포인터 한 줄**만 갱신한다.
