import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

def get_llm(temperature=0.3):
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(
            model_name="openai/gpt-oss-120b",
            groq_api_key=groq_key,
            temperature=temperature,
        )
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
        max_retries=10,
        timeout=120,
    )

def format_docs(docs):
    print(f"\n--- DEBUG: Retrieved {len(docs)} Chunks ---")
    for i, doc in enumerate(docs):
        print(f"[{i}]: {doc.page_content[:100]}...")
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 4)

    llm = get_llm()

    prompt_template = """You are an intelligent, helpful AI Video & Meeting Assistant.
Your goal is to provide clear, insightful, and comprehensive answers to the user's question based on the provided meeting/video transcript context.

Guidelines:
1. **Synthesize & Explain**: Provide a well-structured, natural, and informative response. If the user types a keyword or topic (e.g., "dream", "libro"), summarize how that topic is discussed in the transcript, providing key details and surrounding context rather than just a single raw quote.
2. **Clear Formatting**: Use clean formatting, bold text, and bullet points where appropriate to make the output easy to read.
3. **Factual Accuracy**: Base your response strictly on the information present in the transcript context. Do not invent details not present in the context.
4. **Fallback**: If the query is completely unrelated or not found in the context, politely state:
"I could not find information regarding your query in the meeting transcript."

Context from meeting transcript:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{question}"),
    ])

    # full LCEL RAG pipeline 
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)

    llm = get_llm()
    prompt_template = """You are an intelligent, helpful AI Video & Meeting Assistant.
Your goal is to provide clear, insightful, and comprehensive answers to the user's question based on the provided meeting/video transcript context.

Guidelines:
1. **Synthesize & Explain**: Provide a well-structured, natural, and informative response. If the user types a keyword or topic (e.g., "dream", "libro"), summarize how that topic is discussed in the transcript, providing key details and surrounding context rather than just a single raw quote.
2. **Clear Formatting**: Use clean formatting, bold text, and bullet points where appropriate to make the output easy to read.
3. **Factual Accuracy**: Base your response strictly on the information present in the transcript context. Do not invent details not present in the context.
4. **Fallback**: If the query is completely unrelated or not found in the context, politely state:
"I could not find information regarding your query in the meeting transcript."

Context from meeting transcript:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    import time
    print(f"Question : {question}")
    for attempt in range(5):
        try:
            answer = rag_chain.invoke(question)
            print(f"answer :{answer}")
            return answer
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                wait = 3 * (2 ** attempt)
                print(f"⚠️ Mistral Rate limit reached. Retrying in {wait}s... (Attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise e
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer
