# s4 piano stem validation

`102 - Dir.wav`의 피아노 사건을 동일 원음에서 귀속하기 위한 독립 실험
작업공간이다. 분리 모델 출력은 정답이나 검출기 입력이 아니라 진단용 참조다.

## Models

- BS-Roformer SW 6-stem: 공개 piano leaderboard의 강한 모델. 체크포인트의
  학습 provenance와 weights license는 미기재이므로 로컬 연구 진단에만 쓴다.
- Spleeter 5-stem: 독립 legacy baseline. 기본 모델의 유효 대역은 11kHz다.
- HTDemucs `htdemucs_6s`: 공식 공개 baseline. 공식 문서도 piano stem의
  bleed와 artifact가 많다고 경고한다.

실제 package 버전, model ID, 체크포인트 SHA-256은
`out/stems/Dir/stem_manifest.json`에 기록된다.

## Workspace

`runtime/`, `models/`, `work/`는 이 디렉터리 아래에 격리하고 git에서 제외한다.
기존 프로젝트 Python 3.14 및 공유 venv는 수정하지 않는다.

## Run

공유 분석 환경에서 다음 순서로 실행한다.

```powershell
python separate.py --models bs_roformer spleeter demucs
python sonify_consensus.py
python verify.py
```

`separate.py --force`는 세 모델 inference부터 다시 수행한다. Canonical FLOAT
WAV는 libsndfile PEAK timestamp를 0으로 고정해 바이트 단위 결정성을 보장한다.

## Listening order

1. `out/stems/Dir/{model}/piano.wav`와 `residual.wav`
2. `전체_stem_{model}_piano_L_residual_R.wav` (`L=piano`, `R=residual`)
3. `전체_stem_{model}_piano_candidate395_클릭.wav`
4. `전체_stem_support395_비교_클릭.wav`
   - 3kHz: 세 모델 모두 지지
   - 5kHz: 세 모델 중 둘이 지지
   - 1.5kHz: 0–1개 모델만 지지
5. `전체_stem_consensus_missed_클릭.wav`

모델 합의도 정답이 아니다. 오르골 음색의 piano 혼입, residual의 피아노 누출,
분리 artifact를 먼저 들은 뒤 사건 지원 수치를 해석해야 한다.
