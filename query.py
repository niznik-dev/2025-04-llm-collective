import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

USE_PIPELINE = False

# Switch between the base model and the fine-tuned model to see differences in output!
MODEL_PATH = f"/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct/"
#MODEL_PATH = f"/scratch/gpfs/CSES/niznik/demo_test_run_1/epoch_0/"

QUERIES = [
    "Linking macrophage metabolism to function in the tumor microenvironment.",
    "Spatial transcriptomics reveals tryptophan metabolism restricting maturation of intratumoral tertiary lymphoid structures.",
    "ZNFX1 Functions as a Master Regulator of Epigenetically Induced Pathogen Mimicry and Inflammasome Signaling in Cancer.",
    "High-Dose Methotrexate in Children and Young Adults With ALL and Lymphoblastic Lymphoma: Results of the Randomized Phase III Study UKALL 2011.",
    "Formulation-Based Cost Savings with Cabozantinib Capsules."
]

def direct_inference(model, tokenizer, queries):
    results = []

    for query in queries:
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.001,
            "do_sample": True
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        response = output[len(prompt_text):].strip()
        results.append(response)

    return results

def pipeline_inference(model, tokenizer, queries):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    results = []
    for query in queries:
        outputs = generator(
            query,
            max_new_tokens=512,
            temperature=0.001,
            do_sample=True,
            return_full_text=False
        )
        results.append(outputs[0]['generated_text'])

    return results

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

if USE_PIPELINE:
    results = pipeline_inference(model, tokenizer, QUERIES)
else:
    results = direct_inference(model, tokenizer, QUERIES)

for query, result in zip(QUERIES, results):
    print(f"Query: {query}")
    print(f"Result: {result}")
    print("")
