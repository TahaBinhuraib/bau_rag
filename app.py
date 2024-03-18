import gradio as gr
import glob
import os
# For checking
# Create Document objects for each text chunk
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
                    ChatPromptTemplate,
                    SystemMessagePromptTemplate,
                    HumanMessagePromptTemplate,
                    )
import deepl

translator = deepl.Translator(os.environ['DEEPL_API_KEY'])

openai.api_key =  os.environ['OPENAI_API_KEY']
system_template = """Use the following pieces of context to answer the users question.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. Also, make sure that the answer is as detailed as possible and use as much information as possible from the context.
                    {context}
                    Begin!
                    ----------------
                    Question: {question}
                    Helpful Answer:"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

all_docs = []  # Reset this for each call
directory = "translated_data"
pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
for files in pdf_files:
    print(files)
    # Read the uploaded file content
    with open(files, "rb") as f:
        loader = PyPDFLoader(str(f.name))
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 400)
        texts = text_splitter.split_text(data[0].page_content)
        docs_one = [Document(page_content=t) for t in texts[:]]
        all_docs.extend(docs_one)

print(len(all_docs))
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai.api_key)
retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs = {"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs = {"prompt": prompt})

def ask(question: str):

    result = qa_chain({"query": question})
    rets = retriever.invoke(question)
    print(rets)
    print(result)
    return result
def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

with gr.Blocks() as demo:
    with gr.Row():
        question_input = gr.Textbox(label="Enter your question")

        submit_button = gr.Button("Submit")
    output = gr.Textbox()

    def process(question):

        question = translator.translate_text(question, target_lang="EN-US").text
        result = ask(question)
        print(result)
        result = translator.translate_text(result['result'], target_lang="TR").text
        return result

    submit_button.click(process, inputs=[question_input], outputs=[output])

demo.launch(share=True)