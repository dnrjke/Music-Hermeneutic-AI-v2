# bs_roformer 피아노 스템 → MIDI 변환 시도 — 계획서

- **작성**: Cursor Grok 4.5
- **일자**: 2026-08-12
- **성격**: 클린 슬레이트 제안 (기존 파이프라인/코드베이스 비참조)
- **범위**: 계획만. 구현·실험 실행은 포함하지 않음
- **소스·조달 보완 (본문 비개정)**: [`piano_midi_sources_v1_cursor_grok_4.5.md`](piano_midi_sources_v1_cursor_grok_4.5.md) — 기존 피아노 자산 재사용·추가 조달 wishlist. 규율 원문: `E:\game\Music Hermeneutic AI\audio\control\README.md`
- **실행 루트**: [`src/exp/s5_midi/clean_amt/`](../src/exp/s5_midi/clean_amt/) (`s4_piano`와 분리)

---

## 1. 목표

`bs_roformer`로 분리된 **피아노 스템(WAV)** 을 입력으로, **청취·편집 가능한 MIDI** 를 만든다.

성공 기준(초안):

| 기준 | 정의 |
|------|------|
| 재생 가능 | DAW/플레이어에서 열리고 피아노 롤이 비어 있지 않음 |
| 시간 정렬 | 원곡 대비 onset이 대략 맞음 (체감 ±50–100ms 이내를 1차 목표) |
| 피치 유용성 | 멜로디/화성의 주요 음이 잡힘 (완전 정확도보다 “편집 시작점”으로 쓸 수 있는지) |
| 재현성 | 동일 입력·동일 설정 → 동일 MIDI (또는 명시된 허용 오차) |

비목표(이번 시도에서 제외):

- 페달·벨로시티·아티큘레이션의 고품질 재현
- 다중 악기 스템 동시 전사
- 상용급 automatic transcription 제품화

---

## 2. 문제 정의

피아노 스템 → MIDI는 단순 “피치 검출”이 아니라 다음이 겹친 문제다.

1. **폴리포니**: 동시에 여러 음이 울림
2. **잔향·겹침**: 분리 스템이라도 어택이 뭉개지거나 다른 악기의 bleed가 남을 수 있음
3. **시간 양자화 vs 표현**: 그리드에 맞추면 듣기 편하지만, 원연주의 루바토/오프셋이 사라짐
4. **옥타브/배음 오류**: 배음을 별도 음으로 잡거나 옥타브를 틀림
5. **지속음(note-off)**: onset만 맞고 duration이 틀리면 MIDI가 “기계음”처럼 들림

`bs_roformer` 피아노 스템은 “깨끗한 솔로 피아노”가 아닐 수 있다는 전제로 설계한다.

---

## 3. 접근 옵션 (후보)

클린 상태에서 우선 검토할 경로. **한 번에 하나를 깊게** 가고, 나머지는 비교용으로만 둔다.

### A. 오프라인 신경망 AMT (권장 1차)

사전학습 Automatic Music Transcription 모델로 WAV → MIDI(또는 note events).

예시 계열(조사·선정 단계에서 확정):

- piano-centric AMT (예: Onsets and Frames 계열, 최신 piano transcription 모델)
- 범용 transcription 중 piano에 강한 모델

**장점**: 폴리포니·duration을 한 번에 다루는 경우가 많음  
**단점**: GPU/의존성, 라이선스, 분리 아티팩트에 대한 강건성 불명

### B. 피치 추적 + 휴리스틱 노트 조립

다중 F0 / salience → peak picking → note clustering → MIDI.

**장점**: 투명하고 디버깅 쉬움  
**단점**: 피아노 폴리포니에서 품질 상한이 낮을 가능성 큼

### C. 하이브리드

신경망 onset/frame + 규칙 기반 post (옥타브 정리, 최소 duration, velocity curve).

**장점**: A의 품질 + B의 통제  
**단점**: 튜닝 파라미터 증가

### 1차 권장

**A를 메인**, 결과가 쓸모없으면 **C로 보강**. B는 베이스라인/진단용.

---

## 4. 입력·출력 계약

### 입력

| 항목 | 제안 |
|------|------|
| 오디오 | `bs_roformer` piano stem WAV (mono 또는 stereo→mono downmix) |
| 샘플레이트 | 모델 권장 SR로 리샘플 (보통 16k/22.05k/44.1k 중 모델 스펙) |
| 길이 | 전체 트랙 또는 짧은 구간(30–90초) 파일럿 |

### 출력

| 산출물 | 설명 |
|--------|------|
| `.mid` | 표준 MIDI file (Format 0 또는 1, 피아노 채널 0/프로그램 0) |
| `notes.json` (선택) | onset, offset, pitch, velocity — 디버그·재현용 |
| `manifest.json` | 모델명/버전, 커밋, 파라미터, 입력 해시, SR, 실행 시각 |
| 청취용 WAV (선택) | MIDI를 동일 템포/길이로 렌더해 원 스템과 A/B |

---

## 5. 파이프라인 스케치

```
piano_stem.wav
    │
    ├─(1) 전처리: mono, peak/RMS 정규화, (선택) mild denoise / HP
    │
    ├─(2) AMT 추론: note events (onset, offset, pitch, vel)
    │
    ├─(3) 후처리:
    │      - 최소 duration / 최소 velocity 컷
    │      - 근접 duplicate merge
    │      - (선택) 템포 추정 후 soft quantize
    │      - (선택) 페달 추정은 보류
    │
    ├─(4) MIDI 직렬화 + manifest
    │
    └─(5) 평가: 청취 A/B + (가능 시) 정량 지표
```

각 단계는 **켜고 끌 수 있는 스위치**로 두어, “모델 단독 vs 후처리 포함”을 분리 비교한다.

---

## 6. 실험 설계

### 6.1 파일럿 (필수)

1. **짧은 구간** 1개 (화성·멜로디가 분명한 30–60초)
2. **어려운 구간** 1개 (페달·밀집 코드·빠른 패세지)
3. 동일 구간을 **모델/설정 2–3종**만 비교 (조합 폭발 금지)

### 6.2 비교 축

| 축 | 예시 |
|----|------|
| 모델 | AMT 모델 A vs B (최대 2) |
| 전처리 | raw vs normalize-only vs light HP |
| 후처리 | none vs merge+min-dur vs +soft quantize |

한 번에 축 하나만 바꾸는 것을 원칙으로 한다.

### 6.3 판정

정량 GT(MIDI ground truth)가 없으면 **청취 루브릭**을 1차로 쓴다.

청취 체크리스트(5점 척도):

1. onset 타이밍
2. 피치/옥타브
3. 화성 밀도(과도한 음 / 빠진 음)
4. note length
5. “편집해서 쓸 수 있는가”

가능하면 나중에 GT 또는 pseudo-GT(수동 수정 MIDI)를 만들어 note-level F1 등으로 보강.

---

## 7. 리스크와 완화

| 리스크 | 완화 |
|--------|------|
| 스템 bleed로 유령 음 | 최소 velocity/에너지 게이트; 실패 시 “피아노 전용 재분리”는 별 트랙으로 분리 |
| 잔향으로 duration 과대 | max duration cap, energy 기반 early note-off |
| 옥타브 오류 | 후처리 옥타브 교정(옵션) + 청취 우선 구간 선정 |
| GPU/환경 의존 | CPU fallback 여부 명시; 컨테이너/venv 고정 |
| 라이선스 | 모델·가중치·코드 라이선스를 선정 전에 표로 정리 |
| 기대 과다 | “자동 완성 MIDI”가 아니라 “스케치 MIDI”로 목표 문구 고정 |

---

## 8. 기술 선정 체크리스트 (구현 전)

구현에 들어가기 전에 아래만 채운다.

- [ ] 후보 모델 2개 이하로 확정 (이름, 논문/레포, 가중치 URL, 라이선스)
- [ ] 필요 Python/CUDA/패키지 버전 pin 초안
- [ ] 입출력 경로·파일 명명 규칙
- [ ] manifest 스키마 초안
- [ ] 파일럿 구간 타임스탬프 (시작–끝 초)
- [ ] 성공/실패 기준을 숫자 또는 루브릭으로 한 줄씩 확정

---

## 9. 제안 디렉터리 레이아웃 (신규, 독립)

기존 구조를 가정하지 않은 **제안**이다. 실제 배치 시 프로젝트 관례에 맞게 조정.

```
experiments/piano_amt/
  README.md                 # 이 계획의 실행 노트
  configs/
    pilot_a.yaml
  scripts/
    transcribe_piano.py     # wav → mid + manifest
    render_midi_preview.py  # mid → wav (선택)
    ab_listen_sheet.md      # 청취 루브릭 기록
  out/
    <run_id>/
      input_hash.txt
      notes.json
      piano.mid
      preview.wav
      manifest.json
```

`run_id` 예: `20260812_grok45_modelX_raw_nopost`

---

## 10. 마일스톤

| 단계 | 산출 | 완료 조건 |
|------|------|-----------|
| M0 선정 | 모델·라이선스·환경 표 | 체크리스트 §8 완료 |
| M1 파일럿 | 짧은 구간 MIDI 1개 | DAW에서 열림 + 청취 기록 |
| M2 비교 | 설정 2–3 런 A/B | 루브릭 점수표 |
| M3 후처리 | merge/min-dur/quantize on/off | “편집 시작점”으로 쓸지 go/no-go |
| M4 정리 | 재현 절차 1페이지 | 동일 입력 재실행 가능 |

go/no-go (M3 종료 시):

- **go**: 파일럿 구간에서 수동 수정이 “처음부터 치는 것”보다 분명 빠름
- **no-go**: 유령 음·옥타브 오류가 심해 수정 비용이 더 큼 → 입력(스템 품질) 또는 모델 교체부터 재검토

---

## 11. 명시적 비범위

- 드럼/보컬/기타 스템 MIDI화
- 실시간/온라인 transcription
- 악보(PDF/MusicXML) 자동 조판 (MIDI 안정화 이후 별도)
- bs_roformer 재학습·파인튜닝
- 기존 프로젝트 내부 유틸과의 통합 (원하면 별 계획)

---

## 12. 다음 액션 (문서만의 다음 단계)

1. §8 체크리스트를 사람이 채울 수 있게 후보 모델 shortlist 조사 (별도 조사 메모)
2. 파일럿용 오디오 구간 지정
3. 구현 착수 여부(go) 결정

이 문서는 **계획**이다. 코드·실험·기존 레포 탐색은 포함하지 않았다.
