from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Load the pre-trained model
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
model.half()

# Ensure model parameters require gradients
for param in model.parameters():
    param.requires_grad = True
    
# Load a dataset suitable for your task (example: NLI dataset)
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev")

# Define a loss function
loss = CosineSimilarityLoss(model)

# Set training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="./models/jina-embeddings-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    args=training_args
)

# Start training
trainer.train()