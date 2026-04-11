<!-- !! 이 파일은 사람이 직접 관리합니다. Claude Code는 이 파일을 참고하거나 수정하지 마세요. !! -->

# TODO

## pi0.5 스타일 Think Annotation 파이프라인 구축

> **목표**: VLA inference 시 observation → **think (subtask reasoning)** → action 흐름을 구현하여,
> VLM이 행동 전에 현재 subtask를 자연어로 추론(think)하고, 그 결과를 conditioning으로 action expert가 action을 생성하는 pi0.5 스타일 아키텍처 완성.

### 참고 자료

- **pi0.5 논문**: [π₀.₅: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- **MaxText** (JAX 기반 LLM 학습 프레임워크): [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext)
- **openpi** (pi0 reference): [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

### 테스트 환경

- **Calvin** (`fywang/calvin-debug-lerobot`, `fywang/calvin-task-ABCD-D-lerobot`)
- **OGBench** (`seohongpark/ogbench`)

---

### 1. Think Annotation 데이터 생성 (MaxText + Gemma 4)

pi0.5는 로봇 trajectory에 semantic subtask label을 annotation하여, 모델이 action 전에 "지금 무엇을 해야 하는지"를 자연어로 예측하도록 학습한다.

- [ ] **MaxText 환경 셋업**: TPU v4-8에서 Gemma 4 모델 로드 (MaxText의 JAX 추론 파이프라인 활용)
- [ ] **Annotation 프롬프트 설계**: Calvin episode의 (image, language_instruction, action_chunk) → Gemma 4에게 "현재 프레임에서 로봇이 수행 중인 subtask를 한 문장으로 설명해줘" 형태의 프롬프트
- [ ] **배치 annotation 스크립트**: 전체 Calvin dataset에 대해 episode별 subtask annotation 생성 → parquet 저장
  - 입력: `(image_top, language_instruction, proprio, raw_actions)`
  - 출력: `think_annotation: str` (e.g., "pick up the red block from the table")
- [ ] **OGBench annotation 확장**: OGBench goal-conditioned 환경에도 동일 파이프라인 적용
- [ ] **Annotation 품질 검증**: 샘플링하여 human eval, 부적절한 annotation 필터링

### 2. 데이터 파이프라인 확장

현재 `VLADataset` protocol에 think annotation 필드 추가.

- [ ] `VLADataset` protocol에 `think: str` 필드 추가 (`data/protocol.py`)
- [ ] `CalvinDataset`에서 annotation parquet 로드 → `__getitem__` 반환값에 `think` 포함
- [ ] OGBench용 dataset 구현 + `@register_dataset("ogbench-*")` 등록

### 3. VLM Think 학습 (2-Stage 확장)

pi0.5의 핵심: VLM이 observation에서 subtask text를 **생성**하도록 학습.

현재 파이프라인:
```
(obs, language) → VLM [frozen] → KV cache → Action Expert → actions
```

목표 파이프라인:
```
Stage 1: (obs, language) → VLM → think tokens (subtask prediction)
Stage 2: (obs, language, think) → KV cache 전처리 → Action Expert → actions
```

- [ ] **VLM think head 추가**: Qwen3-VL의 language model head를 활용하여 think token 생성 (autoregressive)
- [ ] **Think 학습**: VLM을 annotation 데이터로 fine-tune — observation이 주어졌을 때 subtask annotation을 생성하도록
  - LoRA 또는 full fine-tune (HBM 예산에 따라 결정)
  - Loss: cross-entropy on think tokens
- [ ] **Think → KV cache 통합**: think token 생성 후, 전체 (obs + language + think) 시퀀스를 KV cache로 전처리
  - `VLMCacher` 확장: `compute()` 시 think annotation도 VLM에 입력하여 cache에 포함

### 4. Action Expert Conditioning 수정

think tokens가 prefix에 추가되므로 action expert의 prefix-LM 구조 확장.

현재:
```
prefix: [proprio_token(1), obs_tokens(112)] → bidirectional
suffix: [action_tokens(50)] → attend to all
```

목표:
```
prefix: [proprio_token(1), obs_tokens(112), think_tokens(N)] → bidirectional
suffix: [action_tokens(50)] → attend to all
```

- [ ] `GemmaActionExpert.forward_joint()` 수정: think embedding을 prefix에 concatenate
- [ ] `build_prefix_kv_cache()` 수정: inference 시 think tokens 포함
- [ ] `VLAPolicy` 수정: think token projection layer 추가 (VLM hidden → action expert d_model)

### 5. Inference 파이프라인 (pi0.5 스타일)

최종 inference flow:

```
Observation (image + language instruction)
    ↓
VLM: think token 생성 (autoregressive, "pick up the red block")
    ↓
KV cache 구축: (obs + language + think) → frozen prefix cache
    ↓
Action Expert: flow matching denoising (10-step Euler)
    ↓
Actions (50, 7)
```

- [ ] `inference.py` 수정: think 생성 → cache 구축 → denoise 순서 구현
- [ ] Think token 시각화: rollout video에 predicted subtask text 오버레이

### 6. 검증 및 평가

- [ ] Calvin debug (10 chunks)에서 think annotation 포함 학습 → baseline 대비 loss 비교
- [ ] Calvin ABCD→D에서 full ablation
- [ ] OGBench 환경에서 goal-conditioned think annotation 효과 측정
- [ ] Think 품질 분석: inference 시 생성된 think text가 실제 subtask와 일치하는지 정성 평가
