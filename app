import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, pipeline

# Load the fine-tuned model and tokenizer
modelf = DistilBertForQuestionAnswering.from_pretrained("./fine-tuned-distilbert-qa")
tokenizerf = DistilBertTokenizerFast.from_pretrained("./tokenaizer")


st.title("FINE-TUNED DISTILBERT QA SYSTEM (SQuAD Dataset)")
st.write("You can input a context and ask questions based on the context.")

# Input for context (larger text area)
context = st.text_area("Enter the context here:", height=300)

# Input for questions (another text area)
questions = st.text_area("Enter the question here:", height=100)

# Button to perform the QA
if st.button("Get Answer"):
    if context and questions:
        
        qa_pipeline = pipeline("question-answering", model=modelf, tokenizer=tokenizerf)
        result = qa_pipeline(question=questions, context=context)
        
        # Display the results
        st.subheader("Answer:")
        st.write(f"**Answer:** {result['answer']}")
        st.write(f"**Confidence Score:** {round(result['score'], 4)}")
        st.write(f"**Start Position:** {result['start']}")
        st.write(f"**End Position:** {result['end']}")
    else:
        st.warning("Please enter both a context and a question.")
