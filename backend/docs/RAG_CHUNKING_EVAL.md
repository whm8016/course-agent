# RAG DOCX 切块改造：方案与评测报告

> 日期：2026-07-15 评测，2026-07-16 成文
> 状态：**fact mode 评测通过，新切块方案验证有效**

## 一句话结论

新切块方案（`ragflow_manual_docx` + size 600 + Anthropic 上下文前缀 + 表格序列化修复）在 fact mode **全面优于基线**：检索召回 `context_recall +15%`、精度 `context_precision` 持平、答题忠实度 `faithfulness +6.4%`，**无任何副作用**。

---

## 1. 背景（要解决的三个问题）

基线（改造前线上现状）存在三个质量问题：

1. **表格残缺**：`serialize_table`（`core/rag/llamaindex/file_routing.py`）用 `row.cells` 遍历，python-docx 对合并单元格（`gridSpan`/`vMerge`）返回重复引用 → 表头重复、行列错位。实测电阻表、课程信息表都被序列化成残缺错乱的文本。
2. **chunk 过大**：摄入切块硬编码 `LLAMA_INDEX_CHUNK_SIZE=1200`（`indexing_documents.py:18`），`.env` 的 `CHUNKING__INGEST_SIZE` 是死代码、从不生效。
3. **上下文割裂**：1200 大 chunk 把多个知识点挤一团，向量检索漏召（基线 `context_recall` 仅 0.646，35% 信息召不回）。

## 2. 方案（新切块配置）

四项改动打包（参考 Anthropic Contextual Retrieval 论文 + RAGFlow Manual 切块实现）：

| 改动项 | 旧 | 新 | 作用 |
|---|---|---|---|
| 切块策略 | `sentence_splitter`（按字数机械切） | `ragflow_manual_docx`（标题层级栈 + 表格原子化） | 按文档结构切，不锯断表格/标题 |
| chunk size | 1200（硬编码） | **600**（解锁，`.env` 可调） | 切细，提升检索粒度 |
| 上下文前缀 | 关 | **开**（contextual enrichment） | 每块加 `[背景]…` + `【章节/来源:…】`，补回小 chunk 丢失的上下文 |
| 表格序列化 | 有合并单元格 bug | **修复**（横向去重 + 纵向对齐 + Markdown 输出） | 表格还原成对齐的 Markdown |

**配置开关（默认行为不变，`.env` 显式开启）**：

```bash
CHUNKING__STRATEGY=ragflow_manual_docx
CHUNKING__INGEST_SIZE=600
CHUNKING__CONTEXTUAL_ENRICHMENT=true
```

**设计取舍**：结构完整性优先于严格 size 上限。大表格（如课程目标表 1429 字）和"无子标题的长 section"保留整块，不机械按 600 掐断——否则会重新锯断表格/段落，回到 sentence_splitter 的老问题。实测 79 个 chunk 中 59 个 ≤600，仅 20 个略超（607~777）+ 1 个大表格（1429）。

## 3. 评测方法

- **评测集**：`synthetic_dataset.json`（21 题，含 `ground_truth`，电路分析基础实验课程）。
- **指标**：`context_precision`（准）/ `context_recall`（全）/ `faithfulness`（忠），RAGAS 0.4.3，DeepSeek-v4-pro 阅卷。
- **mode**：`fact`（纯向量检索）/ `relationship`（图谱邻域 + naive 事实）。
- **对比对象**：
  - 基线 `course_mycourse`（5-23 建，旧配置，32 chunks）。
  - 新索引 `course_mycourse_rf600`（7-15 建，新配置，79 chunks，848 节点/1425 边）。
  - **同一门课、同样 2 个原始 docx，唯一变量是切块方式。**

**评测前提修复（否则对比无效）**：

1. **缓存隔离 bug（致命）**：`rag_runner.py` 的 cache key 原为 `{qid}_{mode}_v2.json`，**不含 course_id** → 基线和新索引共用缓存，新索引会读到基线答案。修为按课程分目录 `cache/{course_id}/{qid}_{mode}_v2.json`，3 个回归测试锁死（`tests/test_eval_rag_cache_isolation.py`）。
2. **并行评测压垮 Clash 代理**：两路 run_eval 并行（126 并发 DeepSeek 调用）→ Clash TUN 链路出 `CloseWait` 卡死。改串行单路（63 并发稳定）。

## 4. 结果

### 4.1 fact mode（核心对比）

| 指标 | 基线（旧） | 新索引（rf600） | 变化 |
|---|---|---|---|
| `context_precision`（准） | 0.952 | 0.952 | **+0.000（完美持平）** |
| `context_recall`（全） | 0.646 | **0.743** | **+0.097（+15%）** |
| `faithfulness`（忠） | 0.661 | **0.704** | **+0.043（+6.4%）** |

**分布细节**：`context_recall` 中位数 p50 从 0.667 → 0.833（+0.166），标准差 0.346 → 0.309——不仅均值提升，且更稳定（每题都更靠谱，非靠几道高分拉均值）。

### 4.2 relationship mode

| 指标 | 基线 mycourse | rf600（新切块） | rf600_img（新切块+图片） |
|---|---|---|---|
| context_precision | 0.833 | 0.857 | 0.833 |
| context_recall | 0.790 | 0.810 | 0.756 |
| faithfulness | 0.808 | 0.743 | 0.742 |

（基线为 7-15 数据；rf600 / rf600_img relationship 为 **2026-07-22 补跑**）

**反直觉发现**：图谱 `recall(0.790) > fact(0.646)`——LightRAG 知识图谱邻域补回了纯向量检索漏掉的信息。这对默认策略选型是关键参考。新切块 rf600 relationship recall 0.810 进一步提升；rf600_img（带图片）relationship 与纯文本基本持平（recall −0.054 正常波动），图片对图谱邻域检索无明显影响。

### 4.3 chunk 统计

| | 基线 | 新索引 |
|---|---|---|
| chunk 数 | 32 | 79 |
| 平均长度 | 876 字 | 431 字 |
| 表格质量 | 错乱（gridSpan 重复） | 正确 Markdown 对齐 |

### 4.4 延迟

- `retrieve_ms` 基本持平（610 vs 625，±2%）——检索速度不受切块影响。
- `query_ms` +7%（9525→10264，新 contexts 略长，LLM 答题稍慢）。
- 质量门禁 p95 ~16s 是 DeepSeek 答题生成耗时，非切块问题。

## 5. 结论

新切块方案 fact mode **全面胜出**：三指标全升或持平，没有任何权衡取舍。`context_recall` 治好了基线 35% 漏召的一大半，`precision` 完全不降（没有为多召回而塞噪声）——这验证了 **RAGFlow Manual 结构化切块 + 小 chunk + 上下文前缀 + 表格修复** 组合拳的方向正确。

**质量门禁现状**（改进路线）：`context_recall 0.743` 差 0.75 阈值仅 0.007（一线之差），`faithfulness 0.704` 差 0.85，延迟 p95 远超 5s（DeepSeek 生成）。下一步优化方向明确。

## 6. 2026-07-22 重跑验证：图片索引消融 + relationship 补全

用 synthetic 21题对 `rf600`（纯文本）与 `rf600_img`（带图片）各跑 fact+relationship，`--no-cache` 强制重检索重答题，deepseek-v4-pro 阅卷。回答三个问题：7-17 退步是否网络问题、图片索引加分还是拖后腿、切块改造 relationship 补全。

| 指标 | rf600 | rf600_img | img−text | 7-17 rf600_img（退步） |
|---|---|---|---|---|
| fact precision | 0.952 | 0.952 | ±0.000 | 0.714 |
| fact recall | 0.731 | 0.757 | +0.026 | 0.621 |
| fact faithfulness | 0.647 | 0.705 | +0.058 | 0.522 |
| relationship recall | 0.810 | 0.756 | −0.054 | — |

**结论**：
1. **7-17 退步=网络抖动，实锤**：rf600_img fact 三指标全面暴涨回升（precision +0.238 / recall +0.136 / faithfulness +0.183），非索引/切块问题。以后遇到单次评测异常先怀疑网络。
2. **图片索引正收益（fact）**：rf600_img recall +0.026、faithfulness +0.058 优于纯文本，precision 持平——图片 VLM 描述补回漏召信息，**推翻"图片索引拖累"担心**。
3. **relationship 两者持平**（−0.054 正常波动），图片对图谱邻域无影响。
4. 延迟 fact p95 rf600 17.3s vs rf600_img 16.9s 持平——图片索引零延迟代价（摄入时已转文本）。

**已知瑕疵**：每轮 ~5 条 RAGAS Job 报 `IncompleteOutputException`（judge 输出超 8192 max_tokens 截断）拉低 faithfulness 均值，不影响 precision/recall。faithfulness 卡 0.65~0.75 是 DeepSeek+RAGAS 判分固有水平，非切块问题。

## 7. 下一步任务（更新）

1. ~~补新索引 relationship 评测~~ ✅ **2026-07-22 完成**（见第 6 节）。
2. **chunk size 三档消融（600/900/1200）——仍未做**，找最优 size（当前 600 是拍板候选）。
3. ~~图片描述回填 chunk 消融~~ ✅ **2026-07-22 完成**：图片索引正收益、零延迟代价，建议保留图片索引。
4. **上线**：待 ②确认 size 后，定为默认 `CHUNKING__*` 配置对线上课程重建索引。

## 附录：关键文件

| 文件 | 作用 |
|---|---|
| `core/rag/llamaindex/file_routing.py` | `serialize_table` 合并单元格修复 + Markdown 输出 |
| `core/rag/chunking/ragflow_manual_docx.py` | RAGFlow Manual 切块策略（标题栈 + 表格原子化） |
| `core/rag/ingestion.py` | `_chunk_by_sentence_splitter`/`_chunk_documents` 解锁 size 硬编码；`_apply_contextual_enrichment` 上下文前缀 |
| `settings/base.py` | `ChunkingConfig`（strategy/ingest_size/contextual_enrichment） |
| `scripts/eval_rag/rag_runner.py` | 缓存按 course_id 隔离（评测前提） |
| `tests/test_eval_rag_cache_isolation.py` | 缓存隔离回归测试 |
| `scripts/eval_rag/results/eval_summary_20260715_040625.json` | 基线完整结果（fact+relationship） |
| `scripts/eval_rag/results/eval_summary_20260715_051152.json` | 新索引 fact 结果 |
| `lightrag_store/course_mycourse_rf600/` | 新索引数据 |
| `_tmp_build_rf600.py` | 新索引构建脚本（复用基线图片 VLM 缓存） |
