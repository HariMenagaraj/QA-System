import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

# explor the model before

qa_model = pipeline("question-answering",model = "distilbert-base-uncased-distilled-squad")

context = """ Hari is boy who is living in coimbatore he has one lapto with 8gb ram. Whith that laptop he is
              working on project which is QA system Tranformar bassed. Now he alredy spent one week to undersdant
              his project now he is doing the project """

questions = [
    "What is ram of hari's laptop?",
    "Where is Hari living?",
    "What project is Hari working on?"
]


result = qa_model(question = questions,context = context)
for r in result:
    print(f"""Answer: '{r['answer']}',
          score: {round(r['score'], 4)},
          start: {r['start']},
          end: {r['end']}""")

output = """Answer: 'QA system Tranformar bassed',
          score: 0.3677,
          start: 138,
          end: 165
      Answer: 'coimbatore',
          score: 0.9923,
          start: 30,
          end: 40
      Answer: 'QA system Tranformar bassed',
          score: 0.9543,
          start: 138,
          end: 165 """

#after

from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, pipeline

# Load the fine-tuned model and tokenizer
modelf = DistilBertForQuestionAnswering.from_pretrained("./fine-tuned-distilbert-qa")
tokenizerf = DistilBertTokenizerFast.from_pretrained("./tokenaizer")



qa_pipeline = pipeline("question-answering", model=modelf, tokenizer=tokenizerf)

context = """ Hari is boy who is living in coimbatore he has one lapto with 8gb ram. Whith that laptop he is
              working on project which is QA system Tranformar bassed. Now he alredy spent one week to undersdant
              his project now he is doing the project """

questions = [
    "What is ram of hari's laptop?",
    "Where is Hari living?",
    "What project is Hari working on?"
]


result = qa_pipeline(question = questions,context = context)
for r in result:
    print(f"""Answer: '{r['answer']}',
          score: {round(r['score'], 4)},
          start: {r['start']},
          end: {r['end']}""")

output = """Answer: '8gb',
          score: 0.2274,
          start: 63,
          end: 66
        Answer: 'coimbatore',
          score: 0.9943,
          start: 30,
          end: 40
        Answer: 'QA system Tranformar bassed',
          score: 0.8869,
          start: 138,
          end: 165 """
