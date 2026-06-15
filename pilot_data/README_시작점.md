# 중국어 CounterSpeech 실험 — 시작점 (Day 0)

## 0. 한눈에 보는 세 데이터의 역할

| 데이터셋 | 규모 | 강점 | 약점 | 본 실험에서의 역할 |
|---|---:|---|---|---|
| **COLD** | 37,480 (train 25,726 / test 5,323 / dev 6,431) | 양이 풍부, 3영역(gender/race/region) 균등, binary 라벨 | 라벨이 거칠다 (0/1) | **혐오 발화 입력 풀** — 영역별 분층 추출 |
| **CCL2025** | 150 (hate 99) | Q1–Q5 4중 어노테이션 (Target+Argument+Group+hateful), lgbt 영역 포함 | 절대 규모 작음 | **금본위(gold) 평가셋** — Stance 평가의 기준 |
| **PANDA** | 785 (hate-CS 542 + 오분류 243) | hate→CS 쌍, 인간 검증 라벨 | hateScore=-1(243)은 오분류로 그대로 복제됨 | **CS 학습 시드 + RQ4 오분류 분석 자료** |

세 데이터는 서로 **다른 일을 한다**. 그래서 합치는 것이 아니라 **역할별로 분리**해서 활용한다.

---

## 1. 가장 중요한 발견 두 가지

### 1-1. CCL2025의 Q1–Q5 어노테이션은 Du Bois의 Stance Triangle과 거의 1:1로 맞다

CCL2025 각 발화는 최대 5개의 (Target, Argument, Group, hateful) 4중 어노테이션을 갖는다.

- **Target** = 누구를 향한 발화인가 → Stance의 **Positioning** 대상
- **Argument** = 어떤 부정적 속성을 부여하는가 → Stance의 **Evaluation** 내용
- **Group** = 보호 범주 (Sexism/Racism/Region/LGBTQ/others) → 영역 분류
- **hateful** = 최종 hate/non-hate 판정

→ CS 생성 후 생성문에 대해 같은 4중 어노테이션을 자동/인간으로 수행하면 **Stance Re-alignment를 정량 측정할 수 있다**. 본 연구 고유 평가축 “Stance Re-alignment”의 측정 도구가 데이터셋 안에 이미 들어 있다.

### 1-2. PANDA의 hateScore = -1 (243건)은 “CS 오분류” 그 자체이다

`PANDA_corpus.xlsx`의 hateScore 분포:
- `+1` (318건): 인간 검증 결과 명확한 혐오 → CS 응답 적절
- `0`  (224건): 모호한 혐오 → CS 응답 적절
- `-1` (243건): **인간 검증 결과 사실은 혐오가 아님** → userEnteredResponse가 원 발화를 그대로 복제

이 243건은 본 연구자의 선행 연구 ② 4.5절 결론(“COLD와 CCL25-Eval10에서 CS 발화가 혐오로 오분류되는 사례가 있다”)의 **직접 증거 사례집**이다. 본 연구 RQ4(오분류 정정)의 학습 데이터로 그대로 사용 가능하다.

---

## 2. 이미 만들어진 파일들

```text
pilot_data/
├── 01_hate_input_pool.csv      # 941건 통합 입력 풀 (COLD+CCL+PANDA hate)
├── 02_ccl_gold_annotation.csv  # CCL2025 4중 어노테이션 보존 (평가 기준)
├── 03_panda_hate_cs_pairs.csv  # PANDA의 진짜 hate→CS 쌍 542건
├── 04_panda_misclassified.csv  # PANDA의 오분류 사례 243건 (RQ4 데이터)
└── 05_pilot_300.csv            # 첫 실험용 249문장 (COLD 150 + CCL hate 99)
```

`05_pilot_300.csv`의 구성:
- COLD에서 영역별 50문장씩 (gender 50, race 50, region 50)
- CCL2025의 hate 99문장 전부 (race 52, gender 19, region 18, **lgbt 10**)
- 합계 249문장 — Day-1 실험을 곧장 시작할 수 있는 규모

---

## 3. Day-1 → Day-7 작업 정의

### Day 1 — 환경 셋업

- GX10에서 vLLM 또는 Ollama를 띄워 Qwen3-32B-Instruct(FP4) 로컬 서빙
- BGE-M3 임베딩 모델 다운로드 (RAG용, 약 2GB)
- Python 환경: `pandas, openpyxl, transformers, vllm, peft, datasets, faiss-cpu`

### Day 2 — Zero-shot 베이스라인

`05_pilot_300.csv`의 249문장 전체에 대해 Qwen3-32B로 Zero-shot CS 생성.

프롬프트:
```
다음 중국어 온라인 혐오표현에 대해, 공격적이지 않고 설득력 있는 중국어 CounterSpeech를 1~2문장으로 생성하라.
조건: (1) 원문 혐오표현을 반복하지 말 것 (2) 특정 집단을 다시 비하하지 말 것
입력: {hate_text}
출력:
```

산출: `outputs/qwen3_32b_zeroshot.csv` (249행, columns: id, hate_text, cs_generated)

### Day 3 — 자동 평가 (1차 필터)

각 생성문에 대해 다음을 측정:
- **재혐오화 검사**: Qwen3Guard 또는 COLD 분류기로 생성문이 다시 hate(label=1)로 분류되는지 (목표: <5%)
- **표면 품질**: 길이 분포, 한자 비율(영문 답변 배제), perplexity

이 단계에서 “생성 자체가 실패”하는 경우를 골라낸다. 보통 zero-shot에서 10~20% 정도의 실패가 발생한다.

### Day 4 — CCL Gold 평가셋으로 Stance Re-alignment 측정

CCL2025의 99 hate 발화에 대해 다음을 비교:
- 원 발화의 Q1–Q5 (Target, Argument, Group)
- 생성 CS에 대해 동일 어노테이션 수행 (Qwen3-72B로 zero-shot 자동 어노테이션 + 본 연구자 검수)

평가:
- **Target 변경률**: CS가 원 Target을 그대로 거론하는가 vs 새로운 행위자/규범으로 옮기는가
- **Argument 부정성**: CS의 Argument가 여전히 부정 평가인가, 중립/긍정으로 재구성되었나
- **Group 보존**: CS가 새로운 보호 범주를 공격하지 않는가

이 단계가 본 연구의 **핵심 차별 지점**이다. 다른 CS 연구는 여기까지 안 본다.

### Day 5 — PANDA를 활용한 Few-shot/RAG 실험

`03_panda_hate_cs_pairs.csv`의 542개 쌍을 RAG 인덱스로 만든다.
- BGE-M3로 임베딩, FAISS HNSW 인덱스
- 신규 hate 입력에 대해 가장 유사한 5쌍 검색 → 프롬프트에 삽입
- 동일 249문장에 대해 다시 생성: `outputs/qwen3_32b_rag.csv`

이때 PANDA의 `generatedResponse1~4` 4개 후보 중 hateScore=+1/0인 것만 RAG 시드로 사용 (오분류 시드 차단).

### Day 6 — Strategy-controlled 생성

세 가지 전략 각각으로 따로 생성:
- 논리 반박: “以偏概全을 지적해 달라”
- 적극적 재구성: “감정은 인정하되 집단 공격은 완화해 달라”
- 사실 응답: “사실/경험으로 반박해 달라”

산출: `outputs/qwen3_32b_strategy_{logical,reinterpret,fact}.csv`

이 시점이면 **같은 249문장에 대해 4가지 조건(zero-shot/RAG/strategy×3) × 1모델 = 6세트**의 CS 생성문이 모인다. 총 1,494개의 CS.

### Day 7 — 첫 인간 평가

249문장 × 4조건 = 996쌍 중에서 영역×조건 균등 분층으로 80쌍 추출.
중국어 모어 화자 1명에게 6항목(비혐오성/관련성/설득력/전략 적합성/자연성/완화 효과) 5점 척도.
Krippendorff’s α는 다음 주에 2번째 어노테이터를 추가하여 측정.

이 시점이면 학회 발표용 1차 결과가 손에 들어온다.

---

## 4. RQ별 데이터 사용 지도

| RQ | 사용 데이터 | 비고 |
|---|---|---|
| RQ1 (LLM이 어떤 유형의 CS를 생성하나) | 01_hate_input_pool 전체 | 다양한 조건에서 분포 분석 |
| RQ2 (영역별 효과적 전략 차이) | 05_pilot_300 (영역 라벨 있음) | CCL의 lgbt 영역도 포함되므로 4영역 분석 가능 |
| RQ3 (3전략 안정 구현) | 02_ccl_gold + Strategy-controlled 출력 | CCL의 Q1–Q5로 정밀 평가 |
| RQ4 (오분류 정정) | 04_panda_misclassified 243건 | binary classifier 학습 데이터 |
| QLoRA 학습 | 03_panda_hate_cs_pairs 542쌍 | hateScore=+1인 318쌍을 우선 사용 권장 |

---

## 5. 즉시 시작 가능한 한 줄 명령

```bash
# Day 1 환경 셋업 (Ollama 가정)
ollama pull qwen3:32b-instruct-q4_K_M
ollama pull bge-m3

# Day 2 baseline 생성 (예시 코드)
python src/02_generate_counterspeech.py \
  --input pilot_data/05_pilot_300.csv \
  --model qwen3:32b \
  --condition zero_shot \
  --output outputs/qwen3_32b_zeroshot.csv
```

---

## 6. 1주차 끝나면 답이 나와 있을 질문

1. Qwen3-32B는 zero-shot으로도 중국어 CS를 “생성은” 하는가? (예/아니오)
2. 영역별 실패율이 다른가? (lgbt가 가장 어려울 가능성)
3. CCL 99 문장에 대한 Stance Re-alignment 점수의 모델 베이스라인은?
4. RAG가 zero-shot 대비 얼마나 개선되는가?
5. PANDA 오분류 243건이 Qwen3Guard로 “비혐오”로 자동 분류되는 비율은?

이 다섯 질문에 대한 답이 첫 학회 발표(M5–M6)의 핵심 결과가 된다.
