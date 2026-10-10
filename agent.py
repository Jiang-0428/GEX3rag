import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag_tool import retrieve_policy_docs

BASE_MODEL_PATH = r"D:\SCSE\1\GEX3\models\qwen3-0.6b"
LORA_ADAPTER_PATH = r"D:\SCSE\1\GEX3\models\specialized_adapter_v3"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH
)
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH
)

def build_rag_prompt(user_question: str, context_list: list[str]) -> str:
    context_text = "\n\n".join(context_list)
    prompt = f"""You are a university IT support assistant.
Answer user's question **only using the provided policy context below**.
If the context does NOT contain the information needed to answer the question, you must reply exactly:
"The requested information is not specified in available guidance. Do not invent facts."

=====Policy Context=====
{context_text}
=====End of Policy Context=====

User Question: {user_question}
Assistant:"""
    return prompt

def rag_answer(user_q:str):
    context_chunks = retrieve_policy_docs(user_q)
    full_prompt = build_rag_prompt(user_q, context_chunks)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    full_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_out.split("Assistant:")[-1].strip()
    return answer, full_prompt

if __name__ == "__main__":
    test_queries = [
        "My friend is visiting campus and cannot use Campus‑Secure. Should I give them my university username and password so they can connect?",
        "What exact phone number should I call for the university IT Service Desk?",
        "I submitted a print job at 4 PM yesterday and forgot to release it. It is 3 PM today. Should it normally still be available?"
    ]
    for q in test_queries:
        print(f"\n=====QUESTION: {q}")
        ans,_ = rag_answer(q)
        print(f"=====ANSWER:\n{ans}")
