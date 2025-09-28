"""Minimal CV Tailor app UI.

- Single heading: "Welcome to CV tailor built by AM"
- Two tall, wide side-by-side input boxes:
    1) Job Description
    2) Current Resume Template
- One tall, editable output textbox below (with Copy button)
- Clear All button under the Tailor Resume button
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from typing import Any

from google import genai
from google.genai import types
import gradio as gr
import utils


logger = logging.getLogger(__name__)

VERTEX_PROJECT = os.getenv("CVTAILOR_VERTEX_PROJECT", "aerobic-layout-442814-a3")
VERTEX_LOCATION = os.getenv("CVTAILOR_VERTEX_LOCATION", "global")
MODEL_NAME = os.getenv("CVTAILOR_MODEL_NAME", "gemini-2.5-flash")

StreamChunk = str | dict[str, Any]


def generate(
    job_description: str,
    current_resume_template: str,
    request: gr.Request
) -> Generator[StreamChunk, None, None]:
  """Stream tailored resume text based on inputs."""
  validate_key_result = utils.validate_key(request)
  if validate_key_result is not None:
    yield validate_key_result
    return

  if not job_description.strip() and not current_resume_template.strip():
    yield "Please provide a job description and resume template to tailor."
    return

  if not job_description.strip():
    yield "Please paste the job description before tailoring your resume."
    return

  if not current_resume_template.strip():
    yield "Please paste your current resume template before tailoring."
    return

  client = genai.Client(
      vertexai=True,
      project=VERTEX_PROJECT,
      location=VERTEX_LOCATION,
  )

  system_instruction = types.Part.from_text(
      text=("You are an expert resume optimization specialist. "
            "Tailor the resume to closely match the job description "
            "(~80–85% alignment) while keeping a professional tone and clean structure. "
            "If needed, synthesize missing but plausible details, and resolve conflicts in favor of relevance. "
            "Return only the revised resume.")
  )

  user_prompt = types.Part.from_text(
      text=(
          "JOB DESCRIPTION:\n"
          f"{job_description}\n\n"
          "CURRENT RESUME TEMPLATE:\n"
          f"{current_resume_template}\n"
      )
  )

  contents = [types.Content(role="user", parts=[user_prompt])]

  config = types.GenerateContentConfig(
      temperature=0.7,
      top_p=0.95,
      seed=0,
      max_output_tokens=65535,
      safety_settings=[
          types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
          types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
          types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
          types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
      ],
      system_instruction=[system_instruction],
  )

  try:
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=config,
    ):
      if not chunk.candidates:
        continue

      candidate = chunk.candidates[0]
      if candidate is None or not candidate.content:
        continue

      parts_as_text = utils.convert_content_to_gr_type(
          candidate.content, use_markdown=True
      )
      for piece in parts_as_text:
        if not piece:
          continue
        yield gr.update(value=piece, append=True)
  except Exception as exc:  # pylint: disable=broad-except
    logger.exception("Failed to generate tailored resume", exc_info=exc)
    yield gr.update(
        value=(
            "We ran into an issue while tailoring the resume. "
            "Please try again in a moment."
        )
    )


def clear_all() -> tuple[str, str, str]:
  """Clears JD, current resume, and output boxes."""
  return "", "", ""


def create_demo() -> gr.Blocks:
  """Create the CV Tailor Gradio interface."""
  with gr.Blocks(theme=utils.custom_theme) as demo:
  # Single heading only
    gr.Markdown("<h2>Welcome to CV tailor built by AM</h2>")

  # Two tall, wide inputs side-by-side
    with gr.Row():
      jd = gr.Textbox(
          placeholder="Job Description",
          lines=28,
          show_label=False,
          elem_id="jd_textbox",
          scale=1,
      )
      current_resume = gr.Textbox(
          placeholder="Current Resume Template",
          lines=28,
          show_label=False,
          elem_id="resume_textbox",
          scale=1,
      )

    # Output textbox below — editable + copy button
    output_resume = gr.Textbox(
        placeholder="Tailored Resume Output",
        lines=26,
        show_label=False,
        elem_id="output_textbox",
        interactive=True,          # <-- editable/interactable
        show_copy_button=True,     # <-- copy-to-clipboard button
    )

    # Controls
    submit_btn = gr.Button("Tailor Resume", variant="primary")
    clear_btn = gr.Button("Clear All")  # shown directly under Tailor Resume

    # Wire up actions
    submit_btn.click(
        fn=generate,
        inputs=[jd, current_resume],
        outputs=output_resume,
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[jd, current_resume, output_resume],
    )

  return demo


demo = create_demo()


if __name__ == "__main__":
  demo.launch(show_error=True)
