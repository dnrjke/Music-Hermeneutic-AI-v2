# Music Hermeneutic AI v2 — 사건 탐지 및 블라인드 검증

**핵심 질문**: 국소 구간에서 뚜렷한 소리(타격, 건반음 등)가 몇 번 발생했는가?
기계가 답하고, 인간 답변과 비교하여 유의미성을 검증한다.

## v1과의 관계

v1은 Bayesian surprise 기반 해석 렌즈 시스템. L_timbre 경로 소진 후 종료.
v2는 근본적으로 다른 접근 — **이산 사건 탐지 + 계수 + 블라인드 검증**.

## 파이프라인

```
오디오 → LUFS 정규화 → STFT → mel → SuperFlux 온셋 포락선
                                          ↓
                              Otsu 임계 → 극대점 → 최소간격 필터 → 이산 사건 목록
                                                                        ↓
                                                    4초 윈도우별 총 계수 → 블라인드 설문
```

## 사용

```bash
python src/pipeline.py                    # 전곡 분석 + 설문 생성
python src/pipeline.py --track cry        # 특정 트랙만
python src/pipeline.py --no-survey        # 설문 없이 계수만
python src/survey_analyse.py resp.json    # 설문 응답 분석
```

## 규율

- **[D-07]** 에너지 계열 금지 — SuperFlux(스펙트럼 변화) 사용
- **[D-18]** 천장(r~0.85)과 바닥(r~0) 사전 정의
- **[D-21]** Otsu 임계(자유 파라미터 0), 최소간격 30ms(물리적 도출)
- **[D-v2-01]** 설문 완료 전 ground_truth.json 열지 않음
- **[D-v2-03]** 윈도우 크기 4초 선언 후 고정

## 파일 구조

```
src/
  config.py          상수, 경로
  audio_io.py        오디오 적재 (v1에서 이식)
  onset.py           SuperFlux 온셋 포락선
  peak_pick.py       Otsu 피크 검출
  counter.py         윈도우별 사건 총 계수
  survey_gen.py      블라인드 설문 클립 생성
  null_model.py      무작위 배치 귀무 모형
  survey_analyse.py  시스템 vs 인간 비교
  pipeline.py        전체 파이프라인
out/                 분석 출력 (.json)
survey/              설문 클립 (.wav) + 템플릿 (.md)
```

## 환경

Python 3.14.2, v1의 .venv 공유. torch/demucs/beat_this 불필요.
