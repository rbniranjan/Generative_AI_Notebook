#Hugging Face Spaces deployment file for a Gradio-based Policy & Claims Agent.

from langchain_community.document_loaders import  PyPDFLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from langchain_huggingface import  HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import getpass
import gradio as gr
import os

PDF_PATH = Path("./Bank.pdf")
HF_KEY = os.getenv("HF_TOKEN", "")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_KEY

def load_pdf():
   # drive.mount('/content/drive')
    #base_path = Path('/content/drive/MyDrive/GenerativeAI')
    #file_path = base_path /'Bank.pdf'
    file_path = str(PDF_PATH)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"PDF loaed and returing page number {len(pages)} \n")
    return pages


def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    return embeddings

def create_vector(chunks, embeddings):
    vector_db = Chroma.from_documents(documents=chunks,
        embedding= embeddings,
        persist_directory="./hf_cloud_db"
    )

    print(f"Created vectore DB using huggig face\n")
    return vector_db

def get_retriver(vector_db):
    retriever = vector_db.as_retriever(search_type="similarity",
        search_kwargs={"k": 5}
    )
    print(f"Retreiver created \n")
    return retriever

def create_model():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        task="text-generation",
        max_new_tokens=500,
        temperature=0.2,
        do_sample=False,
    )

    chat_model = ChatHuggingFace(llm=llm)
    print(f"Model created")
    return chat_model

def build_policy_prompt():
    prompt = ChatPromptTemplate.from_template("""
    You are a Policy & Claims agent.

    Answer the user's question using ONLY the retrieved policy context.

    Rules:
    1. Do not guess.
    2. Do not use outside knowledge.
    3. If the answer is not found in the context, say:
        "Answer not found in the provided policy document."
    4. Keep the answer short and clear.
    5. Mention source page number when available.

    Question:
    {input}

    Context:
    {context}
    """)
    return prompt

def build_claim_prompt():
    prompt = ChatPromptTemplate.from_template("""
You are a Policy & Claims Copilot.

You are doing a claim pre-check, not final claim approval.

Use ONLY the retrieved policy context.

Rules:
1. Do not guess.
2. Do not use outside knowledge.
3. If evidence is missing, say:
   "Unclear based on provided policy context."
4. Give output in this format:

Pre-check Result:
- Likely Covered / Likely Not Covered / Unclear

Reason:
- Short explanation from the policy context

Waiting Period / Limits:
- Mention only if found

Documents Needed:
- Mention only if found

Disclaimer:
- This is only a pre-check, not final claim approval.

User Claim Scenario:
{input}

Context:
{context}
""")
    return prompt

def rag_chaining(retriever, chat_model, prompt):
    document_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain

def chat_function(message, history, mode):
    if mode in ["Q&A", "Policy Q&A"]:
        return ask_policy_question(message, APP["policy_chain"])
    else:
        return claim_precheck(message, APP["precheck_chain"])

def format_sources(docs):
    if not docs:
        return "No sources found."

    lines = []
    seen = set()

    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "N/A")
        key = (source, page)

        if key not in seen:
            seen.add(key)
            lines.append(f"- File: {source}, Page: {page}")

    return "\n".join(lines)

def ask_policy_question(message, policy_chain):
    result = policy_chain.invoke({"input": message})
    answer = result.get("answer", "")
    docs = result.get("context", [])
    sources = format_sources(docs)

    return f"""{answer}

Sources:
{sources}
"""

def claim_precheck(message, precheck_chain):
    result = precheck_chain.invoke({"input": message})
    answer = result.get("answer", "")
    docs = result.get("context", [])
    sources = format_sources(docs)

    return f"""{answer}

Sources:
{sources}
"""
def launch_ui():
    mode_input = gr.Radio(
        choices=["Q&A", "Claim Pre-check"],
        value="Q&A",
        label="Mode"
    )

    demo = gr.ChatInterface(
        fn=chat_function,
        additional_inputs=[mode_input],
        title="Policy & Claims Agent",
        description=(
            "Ask policy questions or run a basic claim pre-check using the uploaded PDF. "
            "Responses are grounded in retrieved document chunks."
        ),
        examples=[
            ["What is covered under hospitalization?", "Q&A"],
            ["What documents are needed to submit a claim?", "Q&A"],
            ["My policy started 4 months ago and I want to claim for a surgery. Is it likely covered?", "Claim Pre-check"],
        ],
        cache_examples=False
    )

    demo.launch(debug=False, share=True)


def initialize_app():
    pages = load_pdf()
    chunks = split_pages(pages)
    embeddings = create_embedding()
    vector_db = create_vector(chunks, embeddings)
    retriever = get_retriver(vector_db)
    chat_model = create_model()

    policy_prompt = build_policy_prompt()
    precheck_prompt = build_claim_prompt()

    policy_chain = rag_chaining(retriever, chat_model, policy_prompt)
    precheck_chain = rag_chaining(retriever, chat_model, precheck_prompt)

    return {
        "policy_chain": policy_chain,
        "precheck_chain": precheck_chain
    }

APP = initialize_app()
demo = launch_ui()

if __name__ == "__main__":
    demo.launch()