# BS piano stem 독립 전사

입력은 `out/stems/Dir/bs_roformer/piano.wav`로 고정한다. Transkun V2를
주 전사 참조로, Basic Pitch를 독립 감사 모델로 사용한다. 전사 confidence나
threshold를 결과에 맞춰 조정하지 않는다.

## 고정 규칙

- Transkun 2.0.1 기본 `2.0.pt`/`2.0.conf`, CUDA 실행
- Basic Pitch 0.4.0 기본 ICASSP 2022 SavedModel과 API 기본 인자
- 원시 note-on은 보존하고 비교용 onset cluster만 30 ms complete-linkage로 생성
- cluster 대표시각은 note-on median
- 주 비교는 ±30 ms 전역 greedy 일대일 매칭
- ±20/50 ms는 민감도 보고만 수행
- pedal과 note offset은 onset 평가에서 제외

## 실행

프로젝트 루트의 PowerShell에서 실행한다.

```powershell
$bp = "src\exp\s4_piano\stem_validation\runtime\venv-basicpitch\Scripts\python.exe"
$project = "E:\game\Music Hermeneutic AI\.venv\Scripts\python.exe"

& $bp "src\exp\s4_piano\stem_validation\transcription\transcribe.py" --determinism-check
& $project "src\exp\s4_piano\stem_validation\transcription\evaluate.py"
& $project "src\exp\s4_piano\stem_validation\transcription\sonify.py"
& $bp "src\exp\s4_piano\stem_validation\transcription\verify.py"
```

전사·평가 결과는 `out/transcription/Dir/`, 청취 파일은
`out/sonify/Dir/transcription/`에 생성된다.

## 해석 제한

두 모델은 같은 BS-Roformer stem을 입력으로 사용하므로 원곡과 완전히 독립된
정답은 아니다. 평가 수치는 true precision/recall이 아니라
`reference coverage`와 `reference-supported fraction`으로만 해석한다.
모델 합의는 신뢰도 레이어이며 최종 사건 귀속은 BS stem 청취로 판단한다.
