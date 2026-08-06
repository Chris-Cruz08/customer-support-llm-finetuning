"""
Gradio app for side-by-side comparison of base Qwen2.5-1.5B-Instruct vs
LoRA fine-tuned version, specialized on the Bitext Customer Support dataset.

Run with: python src/app.py
"""

import os

os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "C:/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "C:/hf_cache/hub"

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "qwen_qlora_adapter")

_tokenizer = None
_base_model = None
_finetuned_model = None


def load_models():
    """
    Loads the base model and the fine-tuned (LoRA adapter) model once at
    startup. Includes basic error handling and validation so failures are
    clear rather than silent.
    """
    global _tokenizer, _base_model, _finetuned_model

    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(
            f"LoRA adapter not found at {ADAPTER_PATH}. "
            "Make sure training has been run and the adapter was saved "
            "(see notebooks/02_finetuning_qlora.ipynb)."
        )

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

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

    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}") from e


def generate_response(model, instruction, max_new_tokens=200):
    """
    Generates a single response from the given model for a user instruction.
    Includes input validation and defensive error handling so the UI never
    crashes on a bad or empty input.
    """
    if not instruction or not instruction.strip():
        return "Please enter a question or request."

    try:
        messages = [{"role": "user", "content": instruction.strip()}]
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id,
            )

        generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = _tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip() if response.strip() else "(No response generated)"

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "Error: GPU ran out of memory. Try a shorter input."
    except Exception as e:
        return f"Error generating response: {e}"


def compare_models(instruction):
    """
    Callback for the Gradio interface. Generates responses from both the
    base and fine-tuned models for the same input, so they can be shown
    side by side.
    """
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
            Enter a customer support style request below to see both responses
            side by side.
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
    print("Loading models, this may take a minute...")
    load_models()
    print("Models loaded. Launching Gradio interface...")

    interface = build_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
    # interface.launch(share=True) #windows tunnel