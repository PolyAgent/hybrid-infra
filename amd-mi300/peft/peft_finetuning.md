Setting up PEFT finetuning

(with config taken from https://rocm.blogs.amd.com/artificial-intelligence/starcoder-fine-tune/README.html)

1. Run the pytorch rocm container -- e.g. `docker run --rm -itd --device /dev/kfd --device /dev/dri --name pytorch-hf --security-opt seccomp=unconfined -v $(pwd):/content rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2`
2. The container has 2 conda envs, py_3.10 and base, which has python v3.12-- the py_3.10 one is the one you want to activate (since not all packages are supported in  py3.12)
3. install BitsAndBytes

```bash
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -S . #Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch
make
pip install .
cd ..
```

install extra libraries-- this is the 'recommended set', but upgrading transformers to the most current version (4.42.0) also works:

```
pip install transformers==4.38.2
pip install peft==0.10.0
pip install deepspeed==0.13.1
pip install accelerate==0.27.2
pip install --upgrade huggingface_hub
pip install wandb==0.16.3
pip install fsspec==2023.10.0
pip install requests==2.28.2
pip install datasets==2.17.1
pip install pandas==2.2.1
pip install numpy==1.22.4
pip install numba==0.59.1
pip install trl==0.9.4
```

An example of finetuning script is [here](./gemma_peft_example.py)