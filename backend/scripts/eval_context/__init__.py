"""上下文预算策略消融评测包（第四批-2）。

对照 arXiv:2508.21433（The Complexity Trap）：raw / masking / summary_only / hybrid 四臂
+ M（keep_recent_turns）扫描，量化各臂对 token 成本、trajectory 轮数、额外压缩 LLM 调用
的影响。复用 Batch 2 已建的 context_policy.set_arm/apply_arm/ARMS（contextvar 覆盖层），
runner 仅 set_arm + 覆盖 settings.context_policy.keep_recent_turns，loop 内自动应用。
"""
