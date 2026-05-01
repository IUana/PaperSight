QA_PROMPT_TEMPLATE = """你是严谨的学术研究员。你只能依据提供的上下文回答，不能使用上下文之外的知识。
如果上下文无法支持结论，请明确写出：{fallback}

上下文：
{context}

问题：{question}

请按以下固定结构输出（Markdown）：
## 结论
- 给出直接结论，并在关键句后标注引用 [C#]

## 流程分步
1. 按顺序列出流程步骤，每一步给出证据 [C#]

## 关键机制
- 说明输入、核心处理、输出，并给出引用 [C#]

## 证据清单
- 列出你使用的关键证据点，格式：`[C#] 证据摘要`

## 局限性
- 明确说明证据不足或不确定处，禁止编造

强约束：
1. 关键结论必须带引用 [C#]
2. 如果证据不足，必须明确写“证据不足”
3. 不要编造实验结果、数字、术语或流程关系
"""


REPORT_PROMPT_TEMPLATE = """你是严谨的学术研究员。你只能依据提供的上下文撰写论文精读报告，不能使用上下文之外的知识。
如果信息不足，请明确写出：{fallback}

上下文：
{context}

请输出 Markdown，并包含以下标题：
1. 研究问题
2. 方法
3. 实验设置
4. 核心结果
5. 局限性
6. 可复现要点

要求：
- 每部分尽量附 [C#] 引用标签
- 不允许出现无证据的推断
"""


FIGURE_PROMPT_TEMPLATE = """你是论文图表解析助手。请仅基于图像内容给出结构化结果，不要臆测。

请输出 JSON，字段必须包含：
- "figure_type": string （如 pipeline / chart / table / other）
- "is_pipeline": boolean
- "confidence": number （0 到 1）
- "steps": string[] （按流程顺序，无则空数组）
- "components": string[]
- "relations": string[] （如 "A -> B"）
- "summary": string （2-4句）
"""


def build_qa_prompt(context: str, question: str, fallback: str) -> str:
    return QA_PROMPT_TEMPLATE.format(context=context, question=question, fallback=fallback)


def build_report_prompt(context: str, fallback: str) -> str:
    return REPORT_PROMPT_TEMPLATE.format(context=context, fallback=fallback)
