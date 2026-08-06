"""
HuggingFace Spaces (ZeroGPU) version of the customer support comparison app.
Uses the @spaces.GPU decorator so HuggingFace allocates a GPU slice per request.
"""

import os
import torch
import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./qwen_qlora_adapter"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Models are loaded lazily, inside a @spaces.GPU-decorated function,
# since ZeroGPU only provides real CUDA access during such calls,
# not at module import / startup time.
_base_model = None
_finetuned_model = None


def _ensure_models_loaded():
    global _base_model, _finetuned_model
    if _base_model is not None and _finetuned_model is not None:
        return

    print("Loading models on first request...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    _base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    finetuned_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    _finetuned_model = PeftModel.from_pretrained(finetuned_base, ADAPTER_PATH)

    print("Models loaded successfully.")


def generate_response(model, instruction, max_new_tokens=150):
    if not instruction or not instruction.strip():
        return "Please enter a question or request."

    try:
        messages = [{"role": "user", "content": instruction.strip()}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip() if response.strip() else "(No response generated)"

    except Exception as e:
        return f"Error generating response: {e}"


@spaces.GPU(duration=90)
def compare_models(instruction):
    _ensure_models_loaded()
    base_output = generate_response(_base_model, instruction)
    finetuned_output = generate_response(_finetuned_model, instruction)
    return base_output, finetuned_output


def build_interface():
    with gr.Blocks(title="Customer Support LLM: Base vs Fine-tuned") as demo:
        gr.Markdown(
            """
            # Customer Support LLM Comparison
            Compare the base **Qwen2.5-1.5B-Instruct** model against a version
            fine-tuned with QLoRA on the Bitext Customer Support dataset.
            Running on HuggingFace ZeroGPU - first request may take a little
            longer while a GPU is allocated.
            """
        )

        with gr.Row():
            instruction_input = gr.Textbox(
                label="Customer request",
                placeholder="e.g. I need help cancelling my order",
                lines=2,
            )

        submit_btn = gr.Button("Compare Responses", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Model (no fine-tuning)")
                base_output = gr.Textbox(label="Response", lines=8, interactive=False)
            with gr.Column():
                gr.Markdown("### Fine-tuned Model (QLoRA)")
                finetuned_output = gr.Textbox(label="Response", lines=8, interactive=False)

        submit_btn.click(
            fn=compare_models,
            inputs=instruction_input,
            outputs=[base_output, finetuned_output],
        )

        gr.Examples(
            examples=[
                "I need help cancelling my order",
                "How do I get a refund for a defective product?",
                "Can I speak to a human representative?",
                "I want to change my shipping address",
            ],
            inputs=instruction_input,
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch()