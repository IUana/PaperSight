QA_PROMPT_TEMPLATE = """你是严谨的学术研究员。
你只能依据提供的上下文回答，不能使用上下文之外的知识。
若上下文无法支持答案，请只回答：{fallback}

上下文：
{context}

问题：
{question}

回答要求：
1. 用清晰中文回答。
2. 引用证据时使用 [C1] [C2] 这样的标签。
3. 不要编造不存在的实验结果、数字或术语。
"""


REPORT_PROMPT_TEMPLATE = """你是严谨的学术研究员。
你只能依据提供的上下文撰写论文精读报告，不能使用上下文之外的知识。
若信息不足，请明确写出：{fallback}

上下文：
{context}

请输出 Markdown，必须包含以下标题：
1. 研究问题
2. 方法
3. 实验设置
4. 核心结果
5. 局限性
6. 可复现要点

每个部分尽量附 [C1] [C2] 引用标签。
"""


def build_qa_prompt(context: str, question: str, fallback: str) -> str:
    return QA_PROMPT_TEMPLATE.format(context=context, question=question, fallback=fallback)


def build_report_prompt(context: str, fallback: str) -> str:
    return REPORT_PROMPT_TEMPLATE.format(context=context, fallback=fallback)
