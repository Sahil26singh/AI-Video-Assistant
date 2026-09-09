# Actionable items , decision , questions 

import time
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

def get_llm(temperature=0.2):
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

def safe_invoke(chain, input_data, retries=5, delay=3):
    """Invoke chain with explicit exponential backoff for rate limits."""
    for attempt in range(retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                wait = delay * (2 ** attempt)
                print(f"⚠️ Mistral Rate limit reached. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise e
    return chain.invoke(input_data)

def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    time.sleep(2)
    return safe_invoke(chain, {"text": transcript})

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    time.sleep(2)
    return safe_invoke(chain, {"text": transcript})

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    time.sleep(2)
    return safe_invoke(chain, {"text": transcript})