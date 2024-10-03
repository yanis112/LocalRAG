#pip install -U autotrain-advanced

#export HF_TOKEN="API_TOKEN"

autotrain llm
--train
--trainer sft
--model meta_llama/Meta-Llama-3-8B-Instruct \
--data-path HuggingFaceH4/no_robots \
--train-split train \
-text-column messages \
--chat-template zephyr \
--mixed-precision bf-16 \
--lr 2e-5 \
--batch-size 4 \
--block-size 1024 \
--padding right \
--username abhishek \
--peft \
--quantization int4 \
--project-name autotrain-my-awesove-llama3 \
--push-to-huf 

#easy app autotraun app --host 127.0.0.1 --port 100000