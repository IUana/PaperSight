# PaperSight（图文增强版 RAG 学术论文精读助手）

PaperSight 是一个本地运行的论文问答系统，基于 RAG（检索增强生成），支持 **文本 + 图表（尤其 Pipeline 图）** 联合理解。

当前版本的核心目标：
- 能识别并利用论文中的流程图回答“方法流程/框架”类问题
- 输出更有深度的结构化答案，并附带可追溯引用

---

## 1. 功能概览

- PDF 上传与去重入库（按文件哈希）
- 文本解析 + 分块（Chunk + Overlap）
- 图像抽取 + 视觉解析（Figure Chunk）
- 图文混合检索（语义 + 关键词）
- 流程类问题增强重排（对 Pipeline 图加权）
- 严格引用溯源（paper_id / page / chunk_id / doc_type / figure_id / confidence）
- 一键论文精读报告（Markdown）
- 历史论文图表后台补跑（幂等）

---

## 2. 技术栈与模型

- 前端：Streamlit
- 检索编排：LangChain
- 向量库：FAISS
- PDF 解析：pypdf
- 文本生成模型（Ollama）：`gemma4:e4b`
- 向量模型（Ollama）：`nomic-embed-text`
- 视觉图表解析模型（Ollama）：`qwen3-vl:4b`

---

## 3. 环境准备

### 3.1 Conda（推荐）

```bash
conda env create -f environment.yml
conda activate papersight
```

### 3.2 或已有环境直接安装

```bash
pip install -r requirements.txt
```

---

## 4. Ollama 模型准备

确保本地 Ollama 已启动，并拉取以下模型：

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama pull qwen3-vl:4b
```

检查模型：

```bash
ollama list
```

---

## 5. 启动应用

```bash
streamlit run app.py
```

---

## 6. 使用流程

1. 上传一篇或多篇 PDF
2. 点击“写入知识库”完成图文入库
3. 选择检索范围（全库 / 单篇）
4. 直接提问
5. 查看答案与引用片段

建议提问示例：
- `请总结这篇论文的 pipeline 流程`
- `方法由哪几个模块组成？`
- `输入到输出的关键步骤是什么？`
- `这个方法的局限性有哪些？`

---

## 7. 回答风格与证据约束

问答默认输出固定结构：
- 结论
- 流程分步
- 关键机制
- 证据清单
- 局限性

并遵循：
- 关键结论必须带 `[C#]` 引用
- 证据不足时明确写“证据不足”
- 不编造实验结果、数字和流程关系

---

## 8. 核心配置（`PaperSightSettings`）

位置：`papersight/core/settings.py`

默认值（关键字段）：
- `chat_model = "gemma4:e4b"`
- `vision_model = "qwen3-vl:4b"`
- `embedding_model = "nomic-embed-text"`
- `enable_figure_parsing = True`
- `max_figures_per_page = 4`
- `pipeline_figure_boost = 0.35`
- `chunk_size = 1000`
- `chunk_overlap = 200`

---

## 9. 核心接口

主类：`papersight.core.service.PaperSightService`

### 9.1 入库

```python
ingest_document(file, title=None) -> {
  paper_id,
  added_chunks,
  added_text_chunks,
  added_figure_chunks,
  deduplicated,
  message
}
```

### 9.2 问答

```python
answer_question(question, scope_mode, paper_id=None, top_k=5) -> {
  answer,
  citations,
  used_chunks
}
```

`citations` 中关键字段：
- `label`
- `paper_id`
- `page`
- `chunk_id`
- `doc_type`（`text` 或 `figure`）
- `figure_id`（当 `doc_type=figure`）
- `confidence`（当 `doc_type=figure`）
- `snippet`

### 9.3 报告

```python
generate_report(paper_id) -> {
  report_markdown,
  citations,
  summary_meta
}
```

---

## 10. 历史论文图表补跑机制

系统启动后会尝试后台执行一次图表补跑：
- 为旧论文补齐 `figure` chunk
- 采用幂等策略（同一 `paper_id + figure_id` 不重复入库）
- 完成后会在 `data/.figure_backfill.done` 写入标记

如需手动触发（代码内）：

```python
service.backfill_figure_chunks()
```

---

## 11. 目录结构

```text
PaperSight/
├─ app.py
├─ requirements.txt
├─ environment.yml
├─ papersight/
│  └─ core/
│     ├─ service.py
│     ├─ pdf_loader.py
│     ├─ vision.py
│     ├─ prompts.py
│     ├─ vector_index.py
│     ├─ catalog.py
│     └─ settings.py
├─ tests/
└─ data/
```

---

## 12. 测试

推荐命令：

```bash
python -m pytest -q tests
```

可选实时集成测试（依赖本地 Ollama Embedding）：

Windows:

```bash
set PAPERSIGHT_RUN_LIVE=1
python -m pytest -q tests/test_integration_live.py
```

Linux/macOS:

```bash
export PAPERSIGHT_RUN_LIVE=1
python -m pytest -q tests/test_integration_live.py
```

---

## 13. 已知限制

- 图像解析依赖 PDF 内嵌图像对象；极端扫描件/特殊编码 PDF 可能抽取不到图。
- 图表语义质量受视觉模型能力影响，建议优先使用清晰图和完整图注。
- 若本地视觉模型不可用，系统会降级为图注/占位信息，不阻塞全文入库。

---

## 14. 许可证

本项目按仓库当前许可证使用。
