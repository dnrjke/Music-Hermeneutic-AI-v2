# s4_piano 기술 자문 보고서 — 피아노 onset 탐지 고도화

> 2026-08-08 · Fable 자문 + Opus 정리
> 대상: `102 - Dir.wav` (dai 작곡, 오르골/피아노, ~117s)
> **피아노 외 케이스에도 참고할 자산으로 보존**

---

## 1. 문제 진단

### 1-1. 기존 도구의 구조적 한계

SuperFlux 기반 onset detection은 하드코어/EDM/드럼 비트에 최적화.
Dir 피아노곡에서 두 가지 문제:

1. **등간격 비트 artifact**: 곡 9초 경부터 저역 비트를 따르는 반복 클릭.
   피아노 사건이 아닌데 baseline(553~618) → no-maxfilter → 모든 intersection으로 전파.
2. **피아노 타건 구조적 누락**: max_filter(size=3)가 인접 mel 빈 에너지로 할인,
   lag=2로 잔향 대비 차이 작음. 새 배음이 기존 배음과 겹치면 flux 소멸.

### 1-2. Fable이 지적한 코드 구조 이슈

- `ONSET_BANDS`가 `low=30-120(킥), mid=120-2000(신디), high=2000-20000(하이햇)` —
  EDM 타악기 전제. 피아노 배음 구조와 불일치.
- `peaks_adaptive`의 게이트·Q1 rescue 모두 SuperFlux 포락선에 의존 —
  포락선이 놓치면 뒤 단계 전부가 누락 상속.

---

## 2. 시도 내역과 결과

| 방식 | 설명 | 피크 수 | 판정 |
|---|---|---|---|
| SuperFlux baseline | 기존 Otsu | 553~618 | 등간격 비트 artifact |
| 2-pass star suppression | Veil 영감, 마스킹→bandpass+norm | 12 신규(유효) | 마스킹 전체 차폐 |
| log1p | SuperFlux에 np.log1p() | 467~696 | 피아노 반응 있으나 미흡 |
| plain flux | lag=1, max_size=1, no detrend | 684 | 등간격 경향 의문 |
| no-maxfilter | lag=2, max_size=1, detrend | 690 | 선율 라인 추적 |
| spectral novelty | 코사인 거리 (진폭 불변) | 1052 | 멜로디 반응, burst 과다 |
| **novelty + local norm (2s)** | bandpass + 99-pct 정규화 + Otsu | **613** | **가장 유망** (baseline과 공통 13개) |
| novelty deburst (100ms gap) | burst 억제 | 549 | burst 해소, 신규 감소 |
| novelty prominence | scipy peak_prominences + Otsu | 491 | 보수적 |
| intersection (novelty × no-maxf 등) | SuperFlux 게이트 | 491~655 | SuperFlux artifact 전파 |
| phase deviation | 위상 편차 단독 | 932 | 음악 무관, **폐기** |
| per-band 독립 Otsu | 저/중/고 각 대역 독립 | 다수 | 오탐, **폐기** |
| log1p + norm (1s/2s/4s) | log1p → bandpass+norm | 577~604 | baseline 90%+ 공통 |
| novelty + norm (1s/2s/4s) | novelty → bandpass+norm | 532~621 | baseline과 거의 무관 |

**현재 최선**: `novelty + local norm (2s)` = 613 peaks.
사용자 청취: "전체_nov_n2s_비교_클릭.wav가 마음에 들었다.
하지만 실제로 건반을 친 모든 순간을 잡아내기엔 미흡."

---

## 3. Fable 자문 전문

### A. novelty + local norm 고도화

#### A-1. 다중 스케일 대역통과 (WTMM 방향)

단일 `smooth_s`가 아닌 디아딕 스케일 사다리(0.25s, 0.5s, 1s, 2s, 4s, 8s).
각 스케일에서 국소정규화 후 Otsu 피크 추출, **스케일 간 정합(chain)되는 극대점만 채택**.

이론적 근거: Mallat & Hwang의 Wavelet Transform Modulus Maxima(WTMM).
진짜 특이점(타건 어택)은 스케일이 커져도 같은 시간 위치에서 극대가 지속.
잡음/artifact는 스케일이 바뀌면 극대 위치가 흔들리거나 소멸.

D-21 호환: "인접 3단계 중 2단계 이상 정합" 같은 고정 기준으로 파라미터 프리 유지.

#### A-2. 블록 경계 아티팩트 제거

현재 타일링(고정 2s/4s 블록)은 블록 경계 부근 이벤트에 불연속 발생.
`scipy.ndimage.percentile_filter`로 **슬라이딩(오버랩) 국소 백분위** 적용.
블록 크기 3종 비교보다 근본적 개선.

#### A-3. 연속값 기하평균 융합

피크 이진화 후 집합 연산(∩, ∪) 대신,
각 novelty 함수를 정규화한 뒤 **연속값 기하평균**으로 합성:

```python
fused = np.sqrt(flux_norm * novelty_norm)
```

두 탐지자가 약하지만 동의하는 이벤트를 이진 AND/OR보다 잘 보존.
최종적으로 한 번만 Otsu 적용.

#### A-4. Otsu 풀링 범위 재고

전체 트랙 풀링 대신 **시간블록별 독립 Otsu** 재시도.
이전 per-band 독립 Otsu(오탐 폐기)와는 축이 다름 (주파수축 → 시간축).
표본 부족 블록은 인접 블록 통계로 폴백.

### B. 근본적으로 다른 접근

#### B-1. 교차대역 동시성(coincidence) 게이트 ★최우선 권고

**물리적 근거**: 피아노 타건은 어느 음역이든 해머-현 타격 순간
**광대역 임펄스성 노이즈**(펠트 타격음)를 반드시 동반.
최저음이라도 어택 순간엔 중고역까지 에너지가 퍼짐.
등간격 저역 비트가 신스/킥이면 어택이 저역에만 집중, 중고역 flux 부재.

**구현**: `band_envelopes()`로 low/mid/high 피크에서
±10ms 내 mid+high 동시 존재 여부를 AND 조건으로 게이트.

**성운 프로젝트 대응**: spectral uniformity(ch_min/ch_max)를 주파수축으로 재매핑.
별(점원)이 여러 색 채널에서 고르게 밝듯,
진짜 타건은 여러 주파수 대역에서 고르게(동시에) flux.
확산원(성운)이 특정 색에 치우치듯, artifact는 저역에 치우침.

per-band 독립 Otsu(OR, 폐기됨)와 달리 **AND 조건** → 특이도 상승.

#### B-2. CQT 기반 novelty

Mel 필터뱅크는 심리음향 등가 대역폭 기준 (EDM 적합).
피아노는 반음 단위 이산 피치 격자 → CQT(Constant-Q Transform)가 적합.
빈 간격이 로그 주파수(반음 비례)라서 화성적 배음 분리 우수.

"새 음의 배음이 울리는 음의 배음과 겹쳐서 max_filter 할인" 문제 자체를 완화.
`librosa.cqt` 위에 동일한 SuperFlux/novelty 로직 적용.
순수 신호처리, 기존 인프라(`onset.py`) 재사용 가능.

#### B-3. Complex-domain flux (Bello 2004)

이전 phase deviation 실패는 위상 단독 사용 때문.
Complex flux = **각 빈의 크기로 가중한** 복소 평면상 예측-실측 유클리드 거리.
크기 + 위상 불연속 동시 반영, 저에너지 빈의 잡음 위상에 안 휘둘림.

레가토/서스테인 페달로 magnitude flux가 약할 때도
현의 재여기로 인한 위상 리셋이 남아 → 현재 누락 케이스에 상보적.

#### B-4. High-Frequency Content (Masri) / 스펙트럼 화이트닝 잔차

HFC(`Σ|X(k)|·k`, 주파수 가중 크기합) — 타격성 어택 특화 고전 온셋 검출자.
또는 LPC 잔차 — 스펙트럼 포락선(공명) 제거 후 화이트닝된 잔차.
해머 임펄스가 배음에 가려지지 않고 선명하게 드러남.

#### B-5. 주기성 + 교차대역 동시성 결합 사후 필터

저역 flux 자기상관으로 주기 T 추정.
"주기적이면서 동시에 저역 전용(광대역 결여)"인 경우만 제거.
두 독립 증거의 AND → 오르골 기계 노이즈나 서브베이스 펄스 안전 제거.

---

## 4. Opus 정리: 두 방향 요약

### 방향 1: novelty+norm 고도화 (점진 개선)

현재 613개 → 더 많은 타건 포착.
핵심 수단: 슬라이딩 percentile_filter, 다중 스케일 정합, 기하평균 융합.
장점: 기존 코드 위에 증분 변경. 단점: SuperFlux artifact 근본 해결 안 됨.

### 방향 2: 새로운 onset function (근본 해결)

SuperFlux를 교체하거나 보완:
- CQT 기반 novelty → 배음 겹침 문제 완화
- 교차대역 동시성 → 등간격 저역 artifact 물리적 분리
- Complex flux → 위상 정보 안전 활용
장점: 문제의 근본 원인 해결. 단점: 새 구현 필요.

### Fable 권고 우선순위

1. **B-1 교차대역 동시성 진단** — 코드 수 줄로 가설 검증
2. **B-2 CQT 전환** — 놓치는 타건의 근본 원인
3. **A-1/A-3 다중 스케일 + 융합** — 점진 개선

### 성운 프로젝트와의 대응

| 성운(Veil Nebula) | 음악(Dir piano) |
|---|---|
| blur(σ=2) - blur(σ=40) 밴드패스 | onset - smooth(onset, 2s) 밴드패스 |
| per-channel 99-pct 정규화 | per-block 99-pct 정규화 |
| spectral uniformity (ch_min/ch_max) | **교차대역 동시성 (low∧mid∧high)** |
| source-derived darkness carving | block-gated adaptive detection |
| conservative star suppression (3%) | 마스킹 없이 local norm |
| 다중 스케일 (DoG) | WTMM 디아딕 스케일 사다리 |

핵심 교훈: 성운 프로젝트의 핵심은 "블러 두 스케일 빼기"가 아니라
**"공간축(다중 스케일) × 색축(교차채널 균일성)"의 이중 판별**.
오디오에서는 **"시간축(다중 스케일) × 주파수축(교차대역 동시성)"**으로 재구성.

---

## 5. 참고

- D-21: 출력 보고 파라미터 고르지 않는다
- D-v2-01: 설문 완료 전 ground_truth.json 열지 않음
- Fable 검증 인프라 제안: 데이터마이닝된 MIDI/악보가 존재하면
  학습 모델이 아닌 참조 정답으로만 사용 가능 (D-21/외부 모델 금지 비저촉)
