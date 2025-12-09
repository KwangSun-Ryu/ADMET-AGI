# ADMET Ontology 성능 평가

## 프로젝트 개요

- **목적**: 신약 개발 초기 단계에서 **ADMET(Absorption, Distribution, Metabolism, Excretion, Toxicity)** 온톨로지 기반 추론을 위한 RDF 그래프 구조 최적화 및 성능 평가
- **구현 내용**: ChemBL(화학 생물 활성) + Gene(유전자 정보) RDF 데이터를 통합하고, **ENN(Entity Normalization Network)**, **CLN(Context Learning Network)**, **CLN-ADMET** 다층 압축 알고리즘을 적용하여 그래프 최적화 성능을 정량 평가

## 모델/런타임

- **RDF 처리**: rdflib (TTL 파싱 및 트리플 스트리밍)
- **그래프 변환**: NetworkX DiGraph (RDF → 계산 가능한 그래프 구조)
- **화학 정보**: RDKit (분자 특성 추출)
- **성능 분석**: pandas DataFrame (압축률, 메트릭 시각화)
- **진행률 모니터링**: tqdm

## 디렉터리 트리 & 파일 설명

```
ADMET Ontology/
  ├─ README.md                                          # 현재 파일 (프로젝트 설명서)
  ├─ ADMET graph computation performance evaluation.ipynb  # 메인 평가 노트북
  └─ Data/                                              # 데이터 디렉터리
     ├─ chembl_36.0_activity.ttl (입력)                 # ChemBL 36.0 원본
     ├─ pc_gene.ttl (입력)                              # Gene 원본
     ├─ chembl_sample_v4.ttl (중간산물)                 # ChemBL 샘플 (6,000 트리플)
     ├─ gene_sample_v4.ttl (중간산물)                   # Gene 샘플 (6,000 트리플)
     └─ merged_v4.ttl (중간산물)                        # 병합된 통합 RDF
```

| 파일 | 설명 |
|------|------|
| `ADMET graph computation performance evaluation.ipynb` | STEP 1~8 파이프라인: 샘플링 → 병합 → 변환 → 정규화 → ENN 압축 → CLN 압축 → CLN-ADMET 최적화 → 성능 평가 |
| `chembl_36.0_activity.ttl` | ChemBL 36.0 화학-활성 관계 RDF (입력, 매우 큼) |
| `pc_gene.ttl` | PubChem Gene 유전자 정보 RDF (입력, 매우 큼) |
| `chembl_sample_v4.ttl` | ChemBL 스트리밍 샘플 (6,000 트리플) |
| `gene_sample_v4.ttl` | Gene 스트리밍 샘플 (6,000 트리플) |
| `merged_v4.ttl` | 병합 결과 RDF 그래프 |

## 실행 및 평가 흐름

1) **데이터 준비**
   - `Data/chembl_36.0_activity.ttl`, `Data/pc_gene.ttl` 준비 (또는 심볼릭 링크)
   - 매우 큰 파일이므로 스트리밍 방식으로 샘플링함

2) **노트북 실행**
   ```bash
   jupyter notebook "ADMET graph computation performance evaluation.ipynb"
   ```
   - 각 셀을 순차적으로 실행 (Shift+Enter)
   - 🚀, 📥, 🎉 이모지로 진행 상황 표시

3) **STEP 1~7: 파이프라인**

   | STEP | 입력 | 프로세스 | 출력 | 역할 |
   |------|------|---------|------|------|
   | 1 | 원본 TTL | 스트리밍 샘플링 (v4.2) | `*_sample_v4.ttl` | 대규모 파일 축소 |
   | 2 | 샘플 RDF | 병합 | `merged_v4.ttl` | ChemBL + Gene 통합 |
   | 3 | 병합 RDF | RDF → NetworkX | DiGraph G | 계산 가능한 형태 변환 |
   | 4 | NetworkX | 정규화 (해시 ID) | DiGraph NG | 노드 ID 단축 |
   | 5 | NG | ENN 압축 (20%) | DiGraph G_ENN | 기본 구조 정리 |
   | 6 | G_ENN | CLN 압축 (45~50%) | DiGraph G_CLN | 문맥 기반 정제 |
   | 7 | G_CLN | CLN-ADMET (20~30%) | DiGraph G_ADMET | 최종 균형 최적화 |

4) **STEP 8: 성능 평가 및 요약**
   - pandas DataFrame으로 원본 vs ENN vs CLN vs CLN-ADMET 비교
   - **압축률(%)**, **노드 수**, **엣지 수**, **파일 크기(KB)** 출력

## 환경/운영 참고

### 시스템 요구사항
- **Python**: 3.8 이상
- **메모리**: 최소 8GB (대규모 RDF 처리용 16GB 권장)
- **디스크**: 최소 10GB (원본 데이터 + 중간 산물)

### 설치

```bash
# 필수 패키지 설치
pip install rdflib networkx rdkit pandas tqdm jupyter

# 노트북 실행
jupyter notebook
```

### 주의사항

- **메모리 제약**: `max_samples` 파라미터로 샘플 크기 조정 가능
- **파일 인코딩**: UTF-8 필수, `errors="ignore"` 설정으로 손상 문자 처리
- **RDKit 경고**: 노트북에서 자동 억제 (`RDLogger.DisableLog`)
- **재현성**: `random.seed(42)` 고정으로 동일 결과 보장

### 압축 알고리즘 요약

| 알고리즘 | 목표 압축률 | 특징 | 용도 |
|---------|----------|------|------|
| **ENN** | 15~25% | Degree 기반 완화 정규화 | 기본 구조 정리 |
| **CLN** | 45~50% | Predicate 빈도 + 노드 차수 가중치 | 문맥 단위 정제 |
| **CLN-ADMET** | 20~30% | Predicate 밀도 + 핵심 노드(상위 35%) 보호 | 최종 균형 상태 |

### 성능 평가 예시

```
┌─────────────┬───────┬────────┬──────────┬────────────┐
│   방식      │ Nodes │ Edges  │ Size(KB) │ 압축율(%)  │
├─────────────┼───────┼────────┼──────────┼────────────┤
│ Original    │ 15,234│ 42,567 │ 1,532.4  │    -       │
│ ENN         │ 15,012│ 34,120 │ 1,228.3  │  19.87%    │
│ CLN         │ 12,456│ 18,905 │  681.5   │  55.62%    │
│ CLN-ADMET   │ 11,789│ 15,234 │  548.4   │  64.25%    │
└─────────────┴───────┴────────┴──────────┴────────────┘
```

### 트러블슈팅

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 메모리 부족 | 대규모 RDF | `max_samples` 감소 |
| RDF 파싱 에러 | 파일 형식 문제 | `format="turtle"` 확인 |
| 느린 처리 | 노드/엣지 과다 | `drop_ratio` 증가 |
| 결과 재현 불가 | 랜덤성 | `random.seed(42)` 확인 |

---

**참고**: ChemBL (https://www.ebi.ac.uk/chembl/), PubChem (https://pubchem.ncbi.nlm.nih.gov/)  
**마지막 업데이트**: 2025년 12월 9일
