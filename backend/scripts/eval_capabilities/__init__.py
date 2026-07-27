"""四能力端到端离线质量评测（基于 Inspect AI）。

跳过 HTTP/前端，直接调用 core.orchestrator.get_orchestrator().handle(ctx)，用 LLM-as-judge
给 chat / quiz / deep_solve / deep_research 四个能力"答得好不好"打分 + 质量门禁。

与 scripts/eval_rag/（RAGAS 评检索质量）互补：eval_rag 评 RAG 内脏，eval_capabilities 评
整只动物跑得怎么样。两者同属 eval 离线依赖组，互不冲突。

设计依据（三维调研）：
  - 多轮可靠性：τ-bench pass^k（arXiv:2406.12045）→ solve/quiz 用 epochs+pass_k_k
  - judge 偏置：异家族 ensemble（arXiv:2410.02736）→ model_graded_qa(model=[...]) 多家族投票
  - 教育维度：TutorBench / GuideEval → chat rubric；EdTech MCQ 7维 → quiz rubric
  - 报告质量：DeepResearch Bench RACE+FACT（arXiv:2506.11763）→ research scorer
"""
