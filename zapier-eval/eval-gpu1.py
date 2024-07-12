#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %pip install -r /content/requirments.txt


# In[ ]:


# Setup environment
import os
from dotenv import load_dotenv
from pprint import pprint as pp

load_dotenv()
print(os.environ['ZAPIER_WEBHOOK'])

import jax
jax.devices()


# In[ ]:


import keras_nlp
# keras.config.set_floatx("float16")
# Model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_7b_en")
gemma_lm.preprocessor.tokenizer = keras_nlp.models.GemmaTokenizer(
    proto="/tokenizer/gemma_ua_ordered.model"
)


# In[ ]:


with open('prompt_examples.txt', 'r') as file:
    test_prompts = [line.rstrip('\n') for line in file]
    
with open('prompt_examples_en.txt', 'r') as file:
    test_prompts_en = [line.rstrip('\n') for line in file]

def process_prompts(test_prompts):
    llm_outputs = []
    for prompt in test_prompts:
        output = gemma_lm.generate(prompt, max_length=96)
        llm_outputs.append(output)
    return llm_outputs


# In[ ]:


import json
def save_to_json(results, output_file_path):
    with open(output_file_path, "w") as outfile:
        for result in results:
            json_record = json.dumps(result)
            outfile.write(f"{json_record}\n")

    print(f"Results saved to {output_file_path}")


# In[ ]:


def sort_filenames_alphanumerically(filenames):
    # Define a function to extract the step value from the filename
    print(filenames)
    def extract_step(filename):
        parts = filename.split('_')
        step_index = parts.index('step') + 1
        return int(parts[step_index])
    
    # Sort the list using the step value as the key
    filenames.sort(key=extract_step)


# In[ ]:


folder_path = "/output1/vanilla_6B_2k_mixed_precision_UA_tokenizer/"
file_names = os.listdir(folder_path)
sort_filenames_alphanumerically(file_names)
results = []
for file in file_names:
    parts = file[:-11].split('_')
    loss = float(parts[parts.index('loss') + 1])
    iter = int(parts[parts.index('step') + 1])
    accuracy = float(parts[parts.index('accuracy') + 1])
    ckpt_path = f"{folder_path}{file}"
    print(ckpt_path)
    gemma_lm.load_weights(ckpt_path)
    print("load done")
    prompt_results = process_prompts(test_prompts)
    prompt_results_en = process_prompts(test_prompts_en)
    print("prompting done")
    result = {
        "file_name": file,
        "loss": loss,
        "iter": iter,
        "accuracy": accuracy,
        "prompt_results": prompt_results,
        "prompt_results_en": prompt_results_en
    }
    save_to_json([result], f"vanilla_6B_2k_mixed_precision_UA_tokenizer/{file[:-11]}.json")
    results.append(result)
