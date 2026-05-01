# PaperSight（RAG 学术论文精读助手）

PaperSight 是一个基于 RAG（检索增强生成）的本地论文阅读助手，帮助你更快完成论文精读：

- 上传 PDF 并构建可持久化知识库
- 支持“全库检索 / 单篇检索”问答
- 回答自动附带可追溯引用片段
- 一键生成结构化精读报告

## 1. 项目功能（MVP）

- PDF 上传与解析
- 文本切块（Chunk + Overlap）
- 向量化与 FAISS 持久化检索
- 严格约束问答（无证据时回退）
- 引用溯源（paper_id / 页码 / chunk_id / 片段）
- 一键精读报告（研究问题、方法、实验设置、核心结果、局限性、可复现要点）

## 2. 技术栈

- 前端：Streamlit
- 编排：LangChain
- 向量库：FAISS
- 生成模型：Ollama `gemma4:e4b`
- 嵌入模型：Ollama `nomic-embed-text`
- PDF 解析：`pypdf`

## 3. Conda 环境准备（推荐）

### 3.1 方式 A：使用 `environment.yml` 一键创建（推荐）

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate papersight
```

### 3.2 方式 B：你已创建 Conda 环境时

如果你已经手动创建了环境（例如 `conda create -n myenv python=3.11`），直接执行：

```bash
conda activate myenv
pip install -r requirements.txt
```

## 4. 模型准备

确保本地 Ollama 已启动，并已拉取模型：

- `gemma4:e4b`
- `nomic-embed-text`

检查命令：

```bash
ollama list
```

## 5. 启动应用

```bash
streamlit run app.py
```

使用流程：

1. 上传一篇或多篇 PDF
2. 点击“写入知识库”
3. 选择检索模式（全库 / 单篇）
4. 输入问题并查看答案与引用
5. 点击“生成一键精读报告”

## 6. 测试

单元测试：

```bash
pytest -q
```

可选实时集成测试（调用本地 Ollama Embedding）：

Windows:

```bash
set PAPERSIGHT_RUN_LIVE=1
pytest -q tests/test_integration_live.py
```

Linux/macOS:

```bash
export PAPERSIGHT_RUN_LIVE=1
pytest -q tests/test_integration_live.py
```

## 7. 目录结构

```text
PaperSight/
├─ app.py
├─ requirements.txt
├─ environment.yml
├─ papersight/core/
├─ tests/
└─ data/
```

## 8. 核心接口

`papersight.core.service.PaperSightService`：

- `ingest_document(file, title) -> {paper_id, added_chunks, deduplicated, message}`
- `answer_question(question, scope_mode, paper_id=None, top_k=5) -> {answer, citations, used_chunks}`
- `generate_report(paper_id) -> {report_markdown, citations, summary_meta}`

## 9. 已知限制

- 仅支持可提取文本的 PDF（扫描版无 OCR 时可能失败）
- 仅本地运行，不包含登录、多租户和远程数据库