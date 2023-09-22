`huggingface-cli login`
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
`python llama.cpp/convert.py ./LLAMA-test/results/llama2/final_merged_checkpoint/ --outtype f16 --outfile ./LLAMA-test/results/llama2/final_merged_checkpoint/ggml-model-f16.bin`
`./llama.cpp/quantize ./results/llama2/final_merged_checkpoint/ggml-model-f16.bin ./results/llama2/final_merged_checkpoint/ggml-model-q4_0.bin q4_0`
