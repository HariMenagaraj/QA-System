# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  
    num_train_epochs=3,  # Reduce to prevent overfitting
    learning_rate=3e-5,  # Slightly medium learning rate
    weight_decay=0.02,  # Increased for better regularization
    logging_dir='./logs',
    logging_steps=1500,
    eval_strategy="steps",  # changed 'eval_strategy'
    eval_steps=1500,
    save_steps=3000,
    load_best_model_at_end=True,
    fp16=True,
    seed=42,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,

)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save_pretrained("./fine-tuned-distilbert-qa"
tokenizer.save_pretrained("./tokenaizer")
