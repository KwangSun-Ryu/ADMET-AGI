# UNIVA 프로젝트 설명서

## 프로젝트 개요

- 목적: 신약 개발 단계(기초 연구·비임상 실험)에서 ADMET(흡수·분포·대사·배설·독성) 예측을 자동화하는 인지·추론형 AGI 에이전트 플랫폼을 구축한다.
- UNIVA 역할: Self-evolving Agent, Generalized ADMET Inference 베이스라인, Toxicity AI 프로토타입을 설계·개발해 능동적 의사결정과 ADMET 전주기 통합 예측 부재 문제를 해소한다.
- 모델/런타임: `25TOXMC_Blowfish_v1.0.9-AWQ`를 VLLM(OpenAI 호환)으로 서비스하고, 평가 스크립트는 `openai`/`AsyncOpenAI` 클라이언트로 호출한다.

## 디렉터리 트리 & 파일 설명

```
Project_Folder/
  ├─ ChemCoTBench/                 # HF dump된 ChemCoTBench 로컬 데이터 (Google Drive에서 다운)
  ├─ 25TOXMC_Blowfish_v1.0.9-AWQ/  # AWQ 양자화 모델 가중치 (Google Drive에서 다운)
  ├─ utils.py                      # JSONL 저장, EM 계산, 비동기 호출, 결과 요약, <think> 제거
  ├─ mmlu_toxic.py                 # 독성 MMLU 평가 스크립트
  ├─ chem_cot.py                   # ChemCoTBench 평가 스크립트
  ├─ mobile_eval_e.py              # Mobile-Eval-E 플랜 생성(로컬 모델) + GPT 채점
  ├─ .env                          # BASE_URL(VLLM), GPT_API_KEY(GPT 채점) 환경 변수
  ├─ mmlu_toxic.json               # 독성 MMLU 문제 200문항
  ├─ run_vlim.sh                   # 서버 실행 파일
  └─ no_tool_chat_template_qwen3.jinja  # Qwen 채팅 템플릿(<think> 처리)
```

## 실행 및 평가 흐름

1) Google Drive에서 압축본 받기

   - `25TOXMC_Blowfish_v1.0.9-AWQ`, `ChemCoTBench` 압축본을 Google Drive에서 먼저 받아 `25ADMET/` 아래에 풀어둔다.
   - Google Drive URL: `https://drive.google.com/file/d/1IWv3Ol3kaWnWGiiWY6tKK3qhgrHBvkZQ/view?usp=drive_link`
   - 압축 해제 후 위치 예시: `Project_Folder/25TOXMC_Blowfish_v1.0.9-AWQ/`, `Project_Folder/ChemCoTBench/ `
2) VLLM 서버 기동

   - `run_vllm.sh`에서 마운트 경로(`-v`), 포트(`-p`), GPU(`CUDA_VISIBLE_DEVICES`)를 환경에 맞게 수정한다.
   - 실행 권한: `chmod +x run_vllm.sh`
   - 서버 실행: `./run_vllm.sh`
   - `.env`에 `BASE_URL`(예: `http://<host>:30002/v1/`), `GPT_API_KEY`를 채운다.
   - 컨테이너에서 결과를 쓰려면 `-v ...:/workspace` 옵션의 `:ro`를 제거하거나 `:rw`로 바꾼다.
3) 독성 MMLU 평가 (`mmlu_toxic.py`)

   * **역할**: Toxicity AI 프로토타입 검증
   * `mmlu_toxic.json`의 `system/prompt/answer`를 메시지로 보내고 EM 점수를 계산한다.
   * 결과를 `MMLU_toxic_results.jsonl`로 저장한다.
4) ChemCoTBench 평가 (`chem_cot.py`)

   * **역할**: Generalized ADMET Inference 보조 검증
   * 로컬 `ChemCoTBench/`를 `datasets.load_from_disk`로 읽고, 시스템 프롬프트로 JSON 포맷 출력을 강제한다.
   * 응답 `"output"`과 `meta`의 `reference`를 비교해 EM을 계산하고 `ChemCoT_results.jsonl`로 저장한다.
5) Mobile-Eval-E 평가 (`mobile_eval_e.py`)

   * **역할**: Self-evolving Agent/플래너 행위 시나리오 검증
   * **액터**: `process_request_vl`이 로컬 VLLM 엔드포인트로 `plan/operations` JSON을 생성한다(필요하면 URL을 현재 VLLM 주소로 수정한다).
   * **채점**: `judge_with_gpt`가 GPT(`GPT_API_KEY` 필요)로 루브릭·참고 동작을 비교해 `rubric_score`, `action_match_score`, `overall_score`를 만든다.
   * 결과를 `MobileEvalE_results.jsonl`로 저장한다.
6) 결과 확인

   - 각 `*_results.jsonl`과 콘솔 `SUMMARY`/평균 점수를 확인한다.

## 환경/운영 참고

- `BASE_URL`은 VLLM OpenAI 엔드포인트를 가리키게 맞춘다. `mobile_eval_e.py` 기본 URL(`http://192.168.0.202:25321/v1/`)이 다르면 수정한다.
- VLLM 옵션(`gpu-memory-utilization`, `tensor-parallel-size`, `max-model-len`)은 GPU 자원에 맞춰 조정한다.
- 응답에 `<think>` 블록이 포함돼도 `run_concurrent_worker`가 제거 후 JSON 파싱을 시도하므로 템플릿을 유지해도 된다.
