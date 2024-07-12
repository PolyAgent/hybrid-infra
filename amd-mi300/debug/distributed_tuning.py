#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2024 Google LLC.

# In[2]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://ai.google.dev/gemma/docs/distributed_tuning"><img src="https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png" height="32" width="32" />View on ai.google.dev</a>
#   <td>
#     <a target="_blank" href="https://www.kaggle.com/code/nilaychauhan/keras-gemma-distributed-finetuning-and-inference"><img src="https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png" height="32" width="70"/>Run in Kaggle</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/google/generative-ai-docs/main/site/en/gemma/docs/distributed_tuning.ipynb"><img src="https://ai.google.dev/images/cloud-icon.svg" width="40" />Open in Vertex AI</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/distributed_tuning.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>

# # Distributed tuning with Gemma using Keras

# ## Overview
# 
# Gemma is a family of lightweight, state-of-the-art open models built from research and technology used to create Google Gemini models. Gemma can be further finetuned to suit specific needs. But Large Language Models, such as Gemma, can be very large in size and some of them may not fit on a sing accelerator for finetuning. In this case there are two general approaches for finetuning them:
# 1. Parameter Efficient Fine-Tuning (PEFT), which seeks to shrink the effective model size by sacrificing some fidelity. LoRA falls in this category and the [Fine-tune Gemma models in Keras using LoRA](https://ai.google.dev/gemma/docs/lora_tuning) tutorial demonstrates how to finetune the Gemma 2B model `gemma_2b_en` with LoRA using KerasNLP on a single GPU.
# 2. Full parameter finetuning with model parallelism. Model parallelism distributes a single model's weights across multiple devices and enables horizontal scaling. You can find out more about distributed training in this [Keras guide](https://keras.io/guides/distribution/).
# 
# This tutorial walks you through using Keras with a JAX backend to finetune the Gemma 7B model with LoRA and model-parallism distributed training on Google's Tensor Processing Unit (TPU). Note that LoRA can be turned off in this tutorial for a slower but more accurate full-parameter tuning.

# ## Using accelerators
# 
# Technically you can use either TPU or GPU for this tutorial.
# 
# ### Notes on TPU environments
# 
# Google has 3 products that provide TPUs:
# * [Colab](https://colab.sandbox.google.com/) provides TPU v2, which is not sufficient for this tutorial.
# * [Kaggle](https://www.kaggle.com/) offers TPU v3 for free and they work for this tutorial.
# * [Cloud TPU](https://cloud.google.com/tpu?hl=en) offers TPU v3 and newer generations. One way to set it up is:
#   1. Create a new [TPU VM](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#tpu-vms)
#   2. Set up [SSH port forwarding](https://cloud.google.com/solutions/connecting-securely#port-forwarding-over-ssh) for your intended Jupyter server port
#   3. Install Jupyter and start it on the TPU VM, then connect to Colab through "Connect to a local runtime"
# 
# ### Notes on multi-GPU setup
# 
# Although this tutorial focuses on the TPU use case, you can easily adapt it for your own needs if you have a multi-GPU machine.
# 
# If you prefer to work through Colab, it's also possible to provision a multi-GPU VM for Colab directly through "Connect to a custom GCE VM" in the Colab Connect menu.
# 
# 
# We will focus on using the **free TPU from Kaggle** here.

# ## Before you begin

# ### Kaggle credentials
# 
# Gemma models are hosted by Kaggle. To use Gemma, request access on Kaggle:
# 
# - Sign in or register at [kaggle.com](https://www.kaggle.com)
# - Open the [Gemma model card](https://www.kaggle.com/models/google/gemma) and select _"Request Access"_
# - Complete the consent form and accept the terms and conditions
# 
# Then, to use the Kaggle API, create an API token:
# 
# - Open the [Kaggle settings](https://www.kaggle.com/settings)
# - Select _"Create New Token"_
# - A `kaggle.json` file is downloaded. It contains your Kaggle credentials
# 
# Run the following cell and enter your Kaggle credentials when asked.

# In[ ]:


# If you are using Kaggle, you don't need to login again.
# !pip install ipywidgets kagglehub
# An alternative way is to set KAGGLE_USERNAME and KAGGLE_KEY in your environment if kagglehub.login() doesn't work for you.

# ## Installation
# 
# Install Keras and KerasNLP with the Gemma model.

# In[ ]:


# ### Set up Keras JAX backend

# Import JAX and run a sanity check on TPU. Kaggle offers TPUv3-8 devices which have 8 TPU cores with 16GB of memory each.

# In[ ]:


import jax


print(f"We should see 4 rocm devices:/n{jax.devices()}")"


# In[ ]:


import os
from datetime import datetime

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation
# overhead
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


# ## Load model

# In[ ]:


import keras
import keras_nlp


# ### Notes on mixed precision training on NVIDIA GPUs
# 
# When training on NVIDIA GPUs, mixed precision (`keras.mixed_precision.set_global_policy('mixed_bfloat16')`) can be used to speed up training with minimal effect on training quality. In most case, it is recommended to turn on mixed precision as it saves both memory and time. However, be aware that at small batch sizes, it can inflate memory usage by 1.5x (weights will be loaded twice, at half precision and full precision).
# 
# For inference, half-precision (`keras.config.set_floatx("bfloat16")`) will work and save memory while mixed-precision is not applicable.

# In[ ]:


# Uncomment the line below if you want to enable mixed precision training on GPUs
# keras.mixed_precision.set_global_policy('mixed_bfloat16')


# To load the model with the weights and tensors distributed across TPUs, first create a new `DeviceMesh`. `DeviceMesh` represents a collection of hardware devices configured for distributed computation and was introduced in Keras 3 as part of the unified distribution API.
# 
# The distribution API enables data and model parallelism, allowing for efficient scaling of deep learning models on multiple accelerators and hosts. It leverages the underlying framework (e.g. JAX) to distribute the program and tensors according to the sharding directives through a procedure called single program, multiple data (SPMD) expansion. Check out more details in the new [Keras 3 distribution API guide](https://keras.io/guides/distribution/).

# In[ ]:


# Create a device mesh with (1, 8) shape so that the weights are sharded across
# all 8 TPUs.
device_mesh = keras.distribution.DeviceMesh(
    (1, 4),
    ["batch", "model"],
    devices=keras.distribution.list_devices())


# `LayoutMap` from the distribution API specifies how the weights and tensors should be sharded or replicated, using the string keys, for example, `token_embedding/embeddings` below, which are treated like regex to match tensor paths. Matched tensors are sharded with model dimensions (8 TPUs); others will be fully replicated.

# In[ ]:


model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in the decoder
# attention layers
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (
    model_dim, None, None)

layout_map["decoder_block.*attention_output.*kernel"] = (
    model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear.*kernel"] = (model_dim, None)


# `ModelParallel` allows you to shard model weights or activation tensors across all devcies on the `DeviceMesh`. In this case, some of the Gemma 7B model weights are sharded across 8 TPU chips according to the `layout_map` defined above. Now load the model in the distributed way.

# In[ ]:


model_parallel = keras.distribution.ModelParallel(
    device_mesh=device_mesh, layout_map=layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)

time_model_load = datetime.now()
print(f"Loading Gemma-7B model, it will take awhile, especially very first time, since ~32Gb needs to be download. time:{time_model_load}")
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_7b_en")
print(f"Gemma-7B loaded, took {datetime.now() - time_model_load}")

# Now verify that the model has been partitioned correctly. Let's take `decoder_block_1` as an example.

# In[ ]:


decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
  print(f'{variable.path:<58}  {str(variable.shape):<16}  {str(variable.value.sharding.spec)}')



print(f"loading small training dataset. time:{datetime.now()}")
import tensorflow_datasets as tfds

imdb_train = tfds.load(
    "imdb_reviews",
    split="train",
    as_supervised=True,
    batch_size=2,
)
# Drop labels.
imdb_train = imdb_train.map(lambda x, y: x)

imdb_train.unbatch().take(1).get_single_element().numpy()


# In[ ]:


# Use a subset of the dataset for faster training.
imdb_train = imdb_train.take(20)


# Perform finetuning using [Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA). LoRA is a fine-tuning technique which greatly reduces the number of trainable parameters for downstream tasks by freezing the full weights of the model and inserting a smaller number of new trainable weights into the model. Basically LoRA reparameterizes the larger full weight matrices by 2 smaller low-rank matrices AxB to train and this technique makes training much faster and more memory-efficient.

# Fine-tune on the IMDb movie reviews dataset.

# Limit the input sequence length to 128 to control memory usage.
gemma_lm.preprocessor.sequence_length = 128
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.summary()
gemma_lm.fit(imdb_train, epochs=1)

time0 = datetime.now()
print("==============================================================")
print(f"Start saving weights, time:{time0}")
# Note that enabling LoRA reduces the number of trainable parameters significantly, from 7 billion to only 11 million.
gemma_lm.save_weights("./model.weights.h5")
time1 = datetime.now()
duration_seconds = (time1 - time0).total_seconds()  # Convert timedelta to total seconds
print(f"model.weights.h5 saved, time:{time1} it took {time1-time0}")
print("==============================================================")
print(f"Model size is ~96Gb so the average saving speed was {96/duration_seconds} Gb/s")


# * Learn how to [get started with Keras Gemma](https://ai.google.dev/gemma/docs/get_started).
# * Learn how to [finetune the Gemma model on GPU](https://ai.google.dev/gemma/docs/lora_tuning).
