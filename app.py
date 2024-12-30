import os
from flask import Flask, render_template, request, jsonify
import PyPDF2
import openai
from langchain.chains import ChatChain
from langchain.llms import OpenAI

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = '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'

# Initialize LangChain model (OpenAI GPT)
llm = OpenAI(temperature=0.7)

# Define a simple function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and document processing
@app.route('/process', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file to the 'uploads' directory
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the uploaded PDF
    document_text = extract_text_from_pdf(file_path)

    # Summarize the document using LangChain (GPT)
    summary_chain = ChatChain(llm=llm)
    summary = summary_chain.run(f"Summarize this document:\n{document_text}")

    return jsonify({'summary': summary})

# Route to handle Q&A interaction
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    document_text = request.form.get('document_text')

    if not question or not document_text:
        return jsonify({'error': 'Question or document text missing'}), 400

    # Use LangChain to answer the question based on the document
    question_chain = ChatChain(llm=llm)
    answer = question_chain.run(f"Answer this question based on the document:\n{document_text}\nQuestion: {question}")

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
