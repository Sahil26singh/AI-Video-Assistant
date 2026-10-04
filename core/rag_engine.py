import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

RAG_SYSTEM_PROMPT = """You are an intelligent, helpful AI Video Assistant.
Your goal is to give clear, insightful, and comprehensive answers to the user's question using the provided video transcript context.

Guidelines:
1. **Synthesize & Explain**: Give a well-structured, natural, and informative response. If the user types a keyword or topic (e.g., "dream", "libro"), summarize how that topic is discussed in the video, with key details and surrounding context, not just a single raw quote.
2. **Clear Formatting**: Use clean formatting, bold text, and bullet points where appropriate.
3. **Transcript First**: Start from what the video actually says. Never present your own added knowledge as something the video said.
4. **Relevance Handling**:
   - **Directly covered**: Answer fully from the video transcript.
   - **Partially or loosely related** (the video touches the topic, mentions it briefly, or covers only part of the question): First give what the video says. Then fill the gaps with accurate general knowledge to give a complete answer. Put the added part under a separate heading such as "**Additional context (not from the video):**" so the user can tell the two apart.
   - **Completely unrelated** (nothing in the transcript connects to the question): Reply only with:
     "I could not find information regarding your query in the video transcript."

Context from video transcript:
{context}"""


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


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
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


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    for attempt in range(5):
        try:
            answer = rag_chain.invoke(question)
            print(f"answer :{answer}")
            return answer
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                wait = 3 * (2 ** attempt)
                print(f"Rate limit reached. Retrying in {wait}s... (Attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise e
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer
