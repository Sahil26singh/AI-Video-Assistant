from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

def run_pipeline(source :str, language :str = "english") -> dict:
    print("starting AI Video Assistant")

    chunks, duration_sec = process_input(source)

    transcript = transcribe_all(chunks,language)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    extracted = extract_all(transcript)
    action_item = extracted["action_items"]
    decisions = extracted["key_decisions"]
    questions = extracted["open_questions"]
    
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Takeaways:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your video via RAG
    print("\nChat with your video (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\nAssistant: {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")