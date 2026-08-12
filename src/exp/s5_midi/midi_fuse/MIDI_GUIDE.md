# midi_fuse — MIDI 파일 안내 (청취용)

폴더: `out/` 아래 각 런.  
공통 규칙: **본선 노트는 전부 유지** · rescue는 **같은 음높이(pitch)가 ±30ms 안에 없으면**만 추가.

---

## 무엇부터 들을까

| 목적 | 파일 |
|------|------|
| **정식 go (30–60s)** | [`out/20260812_midi_fuse_clip_harmonic/piano.mid`](out/20260812_midi_fuse_clip_harmonic/piano.mid) |
| **전곡 버전** (같은 방법론) | [`out/20260812_midi_fuse_piano_harmonic_full/piano.mid`](out/20260812_midi_fuse_piano_harmonic_full/piano.mid) |
| 60–90만 바로 재생 | [`…/piano_harmonic_full/piano_t60_90_listen_t0.mid`](out/20260812_midi_fuse_piano_harmonic_full/piano_t60_90_listen_t0.mid) |
| 링잉·사건 연구용 (비급) | `piano_base_only.mid` / `piano_base_t60_90_listen_t0.mid` — **harmonic 없는** piano Transkun |

### harmonic·부가를 왜 넣었나 (채택 이유)

1. **Transkun mid 특성**: 노트 **길이가 짧은** 편. 다른 기법/층을 얹는 목적에 **길이·밀도 보완**(rescue의 한 형태)이 있다.
2. **harmonic**: 이전에도 **rescue** + **노트 길이 등 보완**으로 씀. “실제 타건/사건이 늘었는지”는 **아직 검토 필요**.
3. **당장의 선택 근거**: **감상용으로 좋았는가** (과학적 사건 입증이 아님).
4. **Basic Pitch 인상** (별 축): 왼손·배음 등을 잡아 **풍성해진** 느낌이 눈에 들어옴 — fuse go 대체가 아니라 참고 인상.
5. 링잉·사건으로 깊게 가려면 **harmonic 없는** piano Transkun(`piano_base_only`) 경로. **지금 초점**은 곡→MIDI **감상**.

`*_listen_t0.mid` = 해당 구간을 **0초부터** 들리게 옮긴 것 (재생헤드를 밀 필요 없음).  
이름에 `listen_t0`이 없는 `piano.mid`는 **원곡 절대시각** (30–60 구간이면 처음 30초는 무음일 수 있음).

---

## 1. clip ⊕ harmonic — go 잠금 (30–60s만)

폴더: `out/20260812_midi_fuse_clip_harmonic/`

| 파일 | 한마디 |
|------|--------|
| **`piano.mid`** | 본선(clip 피아노 Transkun) + HPSS harmonic rescue. **청취·후속 작업의 기준 mid.** |

- 구간: 원곡 **30–60초**만 (파일 안 시각도 절대시각 ≈ 30부터).
- 노트 수: 491 (본선 유지 + harmonic에서 37개 추가).
- **채택**: Transkun **짧은 노트** 보완(길이·밀도 rescue) + 청취상 풍성함. **실제 사건 증가 = 미검토.** 당장 근거는 **감상**.

---

## 2. clip ⊕ synthesis — 대조 (30–60s)

폴더: `out/20260812_midi_fuse_clip_synthesis/`

| 파일 | 한마디 |
|------|--------|
| **`piano.mid`** | 본선은 같고 rescue만 **LPC synthesis**. go 아님 · harmonic과 A/B용. |

---

## 3. piano ⊕ harmonic — 풀 길이 (같은 방법론)

폴더: `out/20260812_midi_fuse_piano_harmonic_full/`

본선 = **풀 스템** 피아노 Transkun · rescue = **풀** HPSS harmonic Transkun.  
(파일럿 go의 “clip” 대신 전곡 piano를 쓴 확장판. 규칙(±30ms)은 동일.)

### 전곡 / 큰 파일

| 파일 | 한마디 |
|------|--------|
| **`piano.mid`** | **전곡 융합본.** 들을 때 메인. (절대시각 0~끝, n≈1753) |
| `piano_base_only.mid` | 융합 **전** — 피아노 Transkun만 (harmonic 추가 없음) |
| `harmonic_rescue_in.mid` | rescue **후보 전체** (harmonic Transkun 입력; 융합에 안 들어간 것도 포함) |

### 구간만 · 처음부터 재생 (`listen_t0`)

| 파일 | 한마디 |
|------|--------|
| `piano_t30_60_listen_t0.mid` | 전곡 융합 중 **30–60**만 · 0초부터 |
| **`piano_t60_90_listen_t0.mid`** | 전곡 융합 중 **60–90**만 · 0초부터 (후반 청취용) |
| `piano_base_t60_90_listen_t0.mid` | 60–90 · **피아노만** (harmonic 추가 전) |
| `harmonic_added_t60_90_listen_t0.mid` | 60–90 · 융합에서 **harmonic이 새로 넣은 음만** |

### 60–90에서 차이 듣는 순서 (추천)

1. `piano_base_t60_90_listen_t0.mid` — 피아노만  
2. `piano_t60_90_listen_t0.mid` — + harmonic rescue  
3. `harmonic_added_t60_90_listen_t0.mid` — 추가분만  

→ 2번이 1번보다 나아 보이는지, 3번이 선율인지 배음/잡음인지 보면 됨.

---

## 이름만으로 구분

| 이름에 있으면 | 의미 |
|---------------|------|
| `clip_harmonic` | 30–60 · **go** |
| `clip_synthesis` | 30–60 · 대조 |
| `piano_harmonic_full` | **전곡** · 같은 규칙 |
| `base_only` / `piano_base` | rescue 없이 본선만 |
| `harmonic_…` / `rescue_in` / `added` | harmonic 쪽 (입력 전체 또는 추가분만) |
| `listen_t0` | 구간 시작 = 파일 0초 |
| `t30_60` / `t60_90` | 그 구간만 잘라 냄 |

---

## 이 폴더에 없는 것

- Basic Pitch mid · 506 스냅 mid → `../clean_amt/out/…basic_pitch…`  
- Transkun 원본 단일 스템 mid → `../clean_amt/out/…transkun…`  
- BP와 합친 실험 mid → clean_amt BP 감사 폴더의 `t60_90/` (midi_fuse 정식 산출 아님)

기술·실행 절차는 [`README.md`](README.md).
