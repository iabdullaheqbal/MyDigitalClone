"""Gradio layout for the MyDigitalClone chat application."""

from __future__ import annotations

from functools import partial

import gradio as gr

from clone.chat.session import ChatSession
from clone.persona import AI_CLONE_BANNER, PERSONA_DESCRIPTION, PERSONA_NAME
from clone.service import CloneService
from clone.types import RetrievalMode
from clone.ui.handlers import clear_conversation, handle_message


def build_ui(service: CloneService, reranker_available: bool = False) -> gr.Blocks:
    """Build the queued chat UI with browser-session state and evidence."""
    with gr.Blocks(title=f"{PERSONA_NAME}'s Digital Clone") as demo:
        gr.Markdown(
            f"# {PERSONA_NAME}'s Digital Clone\n"
            f"**{AI_CLONE_BANNER}** {PERSONA_DESCRIPTION}"
        )
        session_state = gr.State(ChatSession(), time_to_live=3600)
        with gr.Row():
            mode = gr.Radio(
                choices=[item.value for item in RetrievalMode],
                value=RetrievalMode.HYBRID_RRF.value,
                label="Retrieval mode",
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Evidence passages")
            rerank = gr.Checkbox(
                value=False,
                label="Cross-encoder reranking",
                interactive=reranker_available,
            )
        chatbot = gr.Chatbot(
            label="Conversation",
            height=440,
            placeholder="Ask a question about the notes.",
        )
        with gr.Row():
            message = gr.Textbox(
                label="Your message",
                placeholder="Ask a question...",
                scale=5,
                lines=1,
            )
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear conversation", scale=1)
        evidence = gr.JSON(label="Retrieved evidence and diagnostics", value={})

        submit = partial(handle_message, service=service)
        inputs = [message, chatbot, session_state, mode, top_k, rerank]
        outputs = [message, chatbot, session_state, evidence]
        send.click(submit, inputs=inputs, outputs=outputs)
        message.submit(submit, inputs=inputs, outputs=outputs)
        clear.click(
            clear_conversation,
            inputs=[],
            outputs=[chatbot, session_state, evidence],
        )
    return demo
