# UNIVA 프로젝트 설명서

## 프로젝트 개요

- **목적**: 신약 개발 초기 단계(기초 연구–비임상 실험)에서 요구되는 ADMET(흡수·분포·대사·배설·독성) 특성 분석을 자동화하기 위해, 인지·추론 기반 AGI 에이전트 플랫폼을 구축한다. 본 플랫폼은 대규모 독성·약물동태 데이터와 온톨로지 기반 지식을 통합하여, 분자 수준의 ADMET 프로파일을 자율적으로 추론·해석할 수 있는 차세대 AI 시스템을 지향한다.
- **구현 내용**: 1차년도 1단계 목표로 Generalized ADMET Inference 베이스라인과 Toxicity AI 프로토타입 모델을 구축한다. 이를 통해 기존 신약개발 과정에서 반복되는 수작업 기반 독성 예측 및 ADMET 전주기 분석의 단절 문제를 해소하고, 능동적 의사결정이 가능한 self-evolving ADMET AI 에이전트 개발의 초석을 마련한다.

## 모델/런타임

- **Inference 파이프라인**

  - OpenAI Chat Completions(Function Calling) 기반 Single-Turn Interaction으로, 필요 시 도구를 여러 번 호출해 컨텍스트를 모은 뒤 최종 답변을 한국어로 생성한다.
  -  `rag_search`: 질의 문자열을 `EMBEDDING_URL`로 임베딩 → OpenSearch k-NN 검색(`OS_HOST/OS_PORT`, 인덱스 `OS_INDEX`, 벡터 필드 `OS_EMBED_FIELD`) → 각 문서의 `embedding` 필드를 제거한 원본 JSON 반환.
  - 기본 설정: `OPENAI_MODEL=gpt-4o-mini`, `RAG_TOP_K=5`, `MAX_TOOL_CALLS=6`; 필요 시 환경 변수로 조정.
- **학습/설계**

  - `Stage1`: 대규모 비라벨 분자 데이터로 자기지도 학습해 Foundation Model(S) 확보
  - `Stage2`: 다수 라벨 데이터셋을 멀티태스크로 지도 학습해 Foundation Model(T)로 고도화
  - `Stage3`: 타깃 속성별 파인튜닝으로 Property-tuned 모델 생성, 멀티모달·문헌·Organoid 소스 데이터까지 연동 가능한 파이프라인 규격을 목표로 설계

## 디렉터리 트리 & 파일 설명

```
Generalized ADMET/
  ├─ inference.py      # OpenSearch k-NN RAG + OpenAI 함수호출 단일 스크립트
  └─ README.md         # 본 문서
```

- `inference.py`: `rag_search` 도구를 포함한 RAG 파이프라인으로, CLI에서 사용자 입력 1회를 받아 최대 `MAX_TOOL_CALLS`까지 반복 도구 호출 후 답변을 생성한다.

## 실행 및 평가 흐름

1) **환경 준비**
   - 의존성 설치:
     ```bash
     pip install opensearch-py openai requests
     ```
   - OpenSearch에 k-NN 인덱스(`OS_INDEX`, 기본 `rag-index`)와 벡터 필드(`OS_EMBED_FIELD`, 기본 `embedding`)를 생성해 둔다.
   - 임베딩 서비스(`EMBEDDING_URL`)는 `{"texts": [...], "batch_size": 64, "normalize": true}` 요청을 받아 `{"embeddings": [[...]]}` 형식으로 응답하도록 준비한다.
2) **환경 변수 설정**
   - OpenAI: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
   - OpenSearch: `OS_HOST`, `OS_PORT`, `OS_INDEX`, `OS_EMBED_FIELD`
   - RAG 동작: `EMBEDDING_URL`, `RAG_TOP_K`(top-k), `MAX_TOOL_CALLS`
3) **실행**
   ```bash
   cd "공개 SW/Generalized ADMET"
   python inference.py
   ```

   - 프롬프트에 질의를 입력하면 시스템 프롬프트 → 도구 호출 → OpenSearch 검색 → 최종 답변 순으로 단일 턴 추론이 수행된다.
4) **로그/결과 확인**
   - 콘솔에 OpenSearch 응답(`embedding` 제거된 JSON)과 도구 호출 로그가 출력된다.
   - 필요 시 `RAG_TOP_K`와 인덱스 필드명을 조정해 검색 품질을 맞춘다.

## 환경/운영 참고

- `OpenSearch`는 SSL/인증 없이 접속하도록 고정돼 있으니 방화벽/VPC 내부에서 구동하거나 별도 인증 계층을 둘 것.
- `OPENAI_API_KEY`는 필수이며, `OPENAI_BASE_URL`이 로컬/VLLM 프록시라면 OpenAI 호환 엔드포인트 규격을 맞춰야 한다.
- `OS_EMBED_FIELD`와 인덱스 매핑의 벡터 차원이 `EMBEDDING_URL` 출력과 일치해야 k-NN 검색이 정상 동작한다.
