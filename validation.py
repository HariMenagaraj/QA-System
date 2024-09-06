from datasets import load_dataset
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, pipeline
import evaluate

# Load the SQuAD dataset
squad_dataset = load_dataset("squad")

# Load your fine-tuned model and tokenizer
modelf = DistilBertForQuestionAnswering.from_pretrained("./fine-tuned-distilbert-qa")
tokenizerf = DistilBertTokenizerFast.from_pretrained("./tokenaizer")

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model=modelf, tokenizer=tokenizerf)

# Dataset splitting
squad_metric = evaluate.load("squad")
validation_set = squad_dataset['validation'] 

# Collect predictions and references
all_predictions = []
all_references = []


for example in validation_set:
    context = example["context"]
    question = example["question"]
    
    # Check if the 'text' list is not empty before accessing element 0
    if len(example["answers"]["text"]) > 0:
        actual_answer = example["answers"]["text"][0]  # The first answer is the reference answer
    else:
        actual_answer = '' # Assign empty string if no answer is available
    
    # Model prediction
    prediction = qa_pipeline(question=question, context=context)['answer']
    
    # Append predictions and actual answers
    all_predictions.append(prediction)
    all_references.append(actual_answer)

import evaluate

# Load the SQuAD evaluation metric
squad_metric = evaluate.load("squad")

def compute_metrics(all_predictions, all_references):
    # Prepare predictions in the required format
    formatted_predictions = [{"id": str(i), "prediction_text": p} for i, p in enumerate(all_predictions)]
    
    # Ensure references contain both "text" and "answer_start" in the correct format
    formatted_references = [
        {"id": str(i), "answers": {"text": [ref], "answer_start": [0]}} 
        for i, ref in enumerate(all_references)
    ]
    
    # Use the SQuAD metric to compute EM and F1
    results = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)
    
    return results['exact_match'], results['f1']

# Compute EM and F1 scores for the predictions
em, f1 = compute_metrics(all_predictions, all_references)


print(f"Exact Match (EM): {em}")
print(f"F1 Score: {f1}")
