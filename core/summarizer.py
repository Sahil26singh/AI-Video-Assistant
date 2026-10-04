import time
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def safe_invoke(chain, input_data, retries=5, delay=3):
    """Invoke chain with explicit exponential backoff for rate limits."""
    for attempt in range(retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                wait = delay * (2 ** attempt)
                print(f"Rate limit reached. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise e
    # Final try
    return chain.invoke(input_data)

def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of the video transcript concisely."),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(2)
        summary_piece = safe_invoke(map_chain, {"text": chunk})
        chunk_summaries.append(summary_piece)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert video content summarizer. Combine these partial summaries "
            "into one final professional video summary in structured bullet points.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()
    time.sleep(2)
    return safe_invoke(combined_chain, {"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the video transcript, generate a short, descriptive, professional video title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    time.sleep(1)
    return safe_invoke(title_chain, {"text": transcript[:2000]})




