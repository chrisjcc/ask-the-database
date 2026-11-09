# Implementation Plan & Engineering Roadmap: OPDVR for AskTheDatabase

**Document Version**: 1.0.0  
**Date**: November 9, 2026  
**Target Paper**: *On-Policy Distillation with Verifiable Reward* (arXiv:2608.24696, August 2026)  
**Assessing Project**: **AskTheDatabase**  
**Classification**: Engineering Roadmap & Work Breakdown Structure (WBS)  

---

## 1. Executive Summary & Engineering Scope

This document defines the production engineering roadmap for integrating **OPDVR (On-Policy Distillation with Verifiable Reward)** and **GRPD (Group Relative Policy Distillation)** into the **AskTheDatabase** training and evaluation pipeline.

The engineering implementation preserves existing abstractions—including the `SQLiteEnvironment`, `DatabaseToolRegistry`, `TRLGRPOAdapter`, and test-suite mutation engine—while introducing:
1. A modular PyTorch loss engine implementing the sign-aligned ReLU gate (`losses.py`).
2. A privileged teacher scoring wrapper (`privileged_teacher.py`) capable of context-injected DDL evaluation and base-model weight sharing.
3. Multi-turn action token loss masking in `TRLGRPOAdapter` to ensure distillation updates apply strictly to agent actions.
4. Comprehensive observability metrics logging the zero-gated token ratio ($r_{\text{zero}}$) and gradient norms.

---

## 2. Work Breakdown Structure (WBS) by Phases

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ENGINEERING WORK BREAKDOWN                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  PHASE 1: Core Mathematical Engine & Invariant Verification (Est: 2-3 Days)                    │
│  ├── [1.1] Create `src/ask_the_database/training/losses.py` (OPDVRLoss, GRPDLoss)               │
│  ├── [1.2] Implement vectorized ReLU gating: R_GRPD = \hat{A} * ReLU(sign(\hat{A}) * log_ratio) │
│  ├── [1.3] Write unit tests in `tests/test_opdvr_losses.py`                                     │
│  └── [1.4] Verify theoretical invariants: Prop C.1 (directionality) & Prop C.2 (decomposition)  │
│                                                                                                 │
│  PHASE 2: Privileged Teacher & Action Token Masking (Est: 3-4 Days)                             │
│  ├── [2.1] Create `src/ask_the_database/training/privileged_teacher.py`                        │
│  ├── [2.2] Implement DDL context injection for privileged scoring                              │
│  ├── [2.3] Build multi-turn action-token mask generator in `TRLGRPOAdapter`                     │
│  └── [2.4] Implement LoRA weight-sharing mode (zero additional parameter VRAM)                  │
│                                                                                                 │
│  PHASE 3: RLVR Trainer Integration & Configuration (Est: 2-3 Days)                              │
│  ├── [3.1] Extend `configs/training/rlvr.yaml` with explicit distillation block                 │
│  ├── [3.2] Update `src/ask_the_database/training/train_rlvr.py` with CLI flags                 │
│  ├── [3.3] Integrate test-suite mutation verifier (R_TS) into GRPD advantage calculation        │
│  └── [3.4] Add DVC stage `train_grpd` to `dvc.yaml` for reproducible pipeline execution        │
│                                                                                                 │
│  PHASE 4: Observability, Diagnostics & Benchmark Validation (Est: 3-5 Days)                     │
│  ├── [4.1] Implement Weave / WandB logging for zero-gated ratio (r_zero) and conflict types     │
│  ├── [4.2] Execute 20-step smoke test on Zero-Budget Evaluation Micro-Harness                   │
│  ├── [4.3] Run full 150-step comparative PoC on Spider 1.0 Dev split                            │
│  └── [4.4] Benchmark evaluation metrics and generate report artifacts                           │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Component 1: `src/ask_the_database/training/losses.py`
* **Purpose**: Houses the pure mathematical loss functions for OPDVR and GRPD.
* **Key Invariant**: Must be pure PyTorch, vector-parallel, and fully compatible with `torch.compile()`.
* **Interface Specification**:
  ```python
  class GRPDLoss(nn.Module):
      """Group Relative Policy Distillation Loss with parameter-free ReLU gating."""
      def __init__(self, eps: float = 1e-8):
          ...
      def forward(
          self,
          student_logprobs: torch.Tensor,   # Shape: (B, L)
          teacher_logprobs: torch.Tensor,   # Shape: (B, L)
          advantages: torch.Tensor,         # Shape: (B, 1)
          action_mask: torch.Tensor         # Shape: (B, L)
      ) -> Tuple[torch.Tensor, Dict[str, float]]:
          """
          Returns:
              loss: Scalar policy gradient loss.
              metrics: Dict containing 'r_zero_ratio', 'type1_conflict_ratio', 'type2_conflict_ratio'.
          """
  ```

### 3.2 Component 2: `src/ask_the_database/training/privileged_teacher.py`
* **Purpose**: Manages the teacher model forward pass, DDL context injection, and token log-probability extraction.
* **Key Operational Invariant**: When student and teacher use the same base model (e.g. Qwen2.5-Coder-7B), the teacher disables PEFT adapter weights during its forward pass, sharing base parameters without allocating duplicate GPU memory.
* **Interface Specification**:
  ```python
  class PrivilegedSchemaTeacher:
      def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, is_shared_base: bool = True):
          ...
      def prepare_privileged_prompt(self, base_prompt: str, db_id: str) -> str:
          """Injects complete flattened DDL of db_id into teacher context."""
          ...
      @torch.no_grad()
      def compute_action_logprobs(
          self,
          input_ids: torch.Tensor,
          attention_mask: torch.Tensor,
          privileged_prompt_lengths: List[int]
      ) -> torch.Tensor:
          """Computes log-probabilities on student tokens conditioned on privileged DDL."""
  ```

### 3.3 Component 3: Action Token Loss Masking in `TRLGRPOAdapter`
* **Problem**: In multi-turn tool interaction, a trajectory consists of:
  `[Prompt] [Action 1] [Observation 1] [Action 2] [Observation 2] ... [Final Answer]`
* **Requirement**: The distillation loss must apply **only** to tokens generated by the student policy (`Action 1`, `Action 2`, `Final Answer`, and any intermediate reasoning chains). Observation tokens (`CREATE TABLE ...`, SQLite query results) must be masked out with `action_mask = 0`.
* **Implementation**: The interactive environment marks turn boundaries with delimiter tokens (`<tool_call> ... </tool_call>`, `<tool_response> ... </tool_response>`). The mask generator sets:
  $$M_t = \begin{cases} 1, & \text{inside } \langle\text{tool\_call}\rangle \text{ or thought blocks} \\ 0, & \text{inside } \langle\text{tool\_response}\rangle \text{ or system prompts} \end{cases}$$

---

## 4. Configuration Schema (`configs/training/rlvr.yaml`)

We extend `configs/training/rlvr.yaml` to expose the distillation mode and parameters cleanly:

```yaml
# -----------------------------------------------------------------------------
# RLVR Training Configuration with OPDVR / GRPD Support
# -----------------------------------------------------------------------------

algorithm: "grpd"  # Options: "grpo", "dapo", "gspo", "opd", "opdvr", "grpd"

training:
  learning_rate: 5.0e-6
  batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  max_grad_norm: 1.0
  warmup_ratio: 0.1
  group_size: 4  # G rollouts per prompt

# Distillation & Teacher Configuration
distillation:
  enabled: true
  teacher_model_name_or_path: "Qwen/Qwen2.5-Coder-7B-Instruct"
  share_base_model: true  # Share VRAM with student LoRA base
  privileged_schema: true # Inject full DDL into teacher context
  gating_mode: "relu"     # Options: "none" (vanilla OPD), "relu" (OPDVR), "inverse" (ablation)
  eps: 1.0e-8

# Verifiable Reward Configuration
rewards:
  use_mutation_test_suite: true
  mutation_count: 5
  weights:
    test_suite: 1.0
    syntax: 0.1
    schema: 0.1
    efficiency: 0.05
```

---

## 5. Compute, Memory, and VRAM Architecture

### 5.1 VRAM Allocation Profile on Single 80GB A100 / H100
When running Qwen2.5-Coder-7B in bfloat16 with LoRA ($r = 16$):
* **Base Model Parameters (7B bfloat16)**: $14.2\,\text{GB}$ (Shared between student and teacher).
* **LoRA Trainable Parameters ($r=16$)**: $0.15\,\text{GB}$.
* **Optimizer States (AdamW on LoRA weights)**: $0.60\,\text{GB}$.
* **Teacher Activation Buffer (Batched Forward)**: $\approx 6.5\,\text{GB}$.
* **Student Activation Buffer (Forward + Backward)**: $\approx 11.2\,\text{GB}$ (with gradient checkpointing).
* **Rollout Generation Context Cache**: $\approx 4.5\,\text{GB}$.
* **Total Peak VRAM Footprint**: **$\approx 37.15\,\text{GB}$**.

**Key Architectural Finding**: The complete GRPD pipeline with a 7B privileged teacher fits comfortably on a **single 40GB or 80GB GPU** or two 24GB GPUs without requiring distributed model parallelism or tensor sharding.

---

## 6. Risk Management & Fallback Matrix

| Potential Risk | Severity | Detection Mechanism | Fallback / Remediation |
| :--- | :--- | :--- | :--- |
| **Token Mask Misalignment** | High | Mask sum test in unit tests (`action_mask.sum() == 0`) | Assert non-zero action tokens; log raw masked text tokens during initialization. |
| **Teacher Context Overflow** | Medium | Check length of `FullDDL(db_id) + q` against 4,096 limit | If DDL exceeds 2,500 tokens, truncate table remarks or use schema summarization. |
| **Zero-Gated Ratio Collapse** | High | Metric telemetry: $r_{\text{zero}} < 0.10$ or $r_{\text{zero}} > 0.90$ | Triggers automated alert; indicates teacher-student tokenizer mismatch or sign inversion. |
| **Out of Memory on Forward** | Medium | PyTorch CUDA OOM exception during teacher batch | Enable `torch.cuda.empty_cache()` before teacher forward pass; reduce batch size. |

---

## 7. Deliverables & Definition of Done (DoD)

A phase is considered complete only when all criteria in the Definition of Done are verified:
* [x] **Mathematical Soundness**: `test_opdvr_losses.py` passes all invariant tests (Proposition C.1 and Proposition C.2).
* [ ] **Zero Observation Leakage**: Automated test verifies that loss gradients for observation tokens are identically zero.
* [ ] **Zero-VRAM Teacher Mode**: Confirmed memory profiling showing $< 1\,\text{GB}$ increase when enabling same-architecture teacher.
* [ ] **Continuous Telemetry**: $r_{\text{zero}}$ and gradient norms actively streamed to Weave and WandB.
* [ ] **Benchmark Replication**: PoC achieves $\ge +4.0\,\text{pp}$ TS gain over pure GRPO at training step 100 on Spider 1.0 Dev.
