from pathlib import Path

import streamlit as st

from papersight import PaperSightService


def render_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("未检索到可引用片段。")
        return

    for citation in citations:
        st.markdown(
            f"**[{citation['label']}]** `{citation.get('title', '')}` | "
            f"page {citation.get('page')} | chunk `{citation.get('chunk_id')}`"
        )
        st.caption(citation.get("snippet", ""))


@st.cache_resource
def get_service() -> PaperSightService:
    return PaperSightService()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "report_payload" not in st.session_state:
        st.session_state.report_payload = None


def main() -> None:
    st.set_page_config(page_title="PaperSight", page_icon="📚", layout="wide")
    init_state()
    service = get_service()

    with st.sidebar:
        st.title("PaperSight")
        st.caption("RAG 学术论文精读助手")
        st.write(f"生成模型：`{service.settings.chat_model}`")
        st.write(f"嵌入模型：`{service.settings.embedding_model}`")

        uploaded_files = st.file_uploader(
            "上传 PDF 论文",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("写入知识库", use_container_width=True):
            if not uploaded_files:
                st.warning("请先上传至少一个 PDF。")
            else:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"正在处理 {uploaded_file.name}..."):
                        result = service.ingest_document(uploaded_file, title=Path(uploaded_file.name).stem)
                    if result["deduplicated"]:
                        st.info(result["message"])
                    elif result["added_chunks"] > 0:
                        st.success(result["message"])
                    else:
                        st.warning(result["message"])
                st.rerun()

        papers = service.list_papers()
        paper_options = {"全库": None}
        for paper in papers:
            paper_options[f"{paper['title']} ({paper['paper_id']})"] = paper["paper_id"]

        selected_paper_label = st.selectbox("论文范围", options=list(paper_options.keys()))
        selected_paper_id = paper_options[selected_paper_label]

        scope_mode = st.radio(
            "检索模式",
            options=["global", "paper"],
            format_func=lambda x: "全库检索" if x == "global" else "单篇检索",
            help="单篇检索会严格限制只从当前选中论文召回片段。",
        )
        if scope_mode == "paper" and not selected_paper_id:
            st.caption("当前未选择具体论文，已自动回退到全库检索。")

        st.divider()
        st.subheader("精读报告")
        report_disabled = selected_paper_id is None
        if st.button("生成一键精读报告", disabled=report_disabled, use_container_width=True):
            with st.spinner("正在生成报告..."):
                st.session_state.report_payload = service.generate_report(selected_paper_id)

    st.title("PaperSight")
    st.caption("上传论文后可直接提问，回答默认附引用片段。")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("查看引用片段"):
                    render_citations(msg["citations"])

    user_question = st.chat_input("请输入关于论文的问题...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        effective_scope = scope_mode
        if scope_mode == "paper" and not selected_paper_id:
            effective_scope = "global"

        with st.chat_message("assistant"):
            with st.spinner("检索与生成中..."):
                result = service.answer_question(
                    question=user_question,
                    scope_mode=effective_scope,
                    paper_id=selected_paper_id if effective_scope == "paper" else None,
                    top_k=service.settings.default_top_k,
                )
            st.markdown(result["answer"])
            with st.expander("查看引用片段"):
                render_citations(result["citations"])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "citations": result["citations"]}
        )

    if st.session_state.report_payload:
        report_payload = st.session_state.report_payload
        st.divider()
        st.subheader("论文精读报告")
        st.markdown(report_payload["report_markdown"])
        with st.expander("报告引用片段"):
            render_citations(report_payload["citations"])
        st.download_button(
            label="下载报告 Markdown",
            data=report_payload["report_markdown"],
            file_name=f"{report_payload['summary_meta'].get('paper_id', 'report')}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
