# Path: phases/11-llm-engineering/06-rag/code/conversation.py
# Lesson: RAG — Retrieval-Augmented Generation
# Conversational RAG with past 3 turns of memory

from collections import Counter
import math
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize the neural embedding model
embedder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed(text):
    return embedder_model.encode(text)

def search(query, stored_embeddings, top_k=3):
    # Calculate similarity scores using sentence-transformers built-in helper
    similarities = embedder_model.similarity(query, stored_embeddings)[0].numpy()
    # argsort returns indices of sorted similarities (ascending)
    top_indices = similarities.argsort()[-top_k:][::-1]
    # Pair indices with their score
    return [(idx, float(similarities[idx])) for idx in top_indices]

def build_rag_prompt(query, retrieved_items, history=None):
    context = "\n\n---\n\n".join(
        f"[Source {item['metadata']['source']}, Chunk {item['metadata']['chunk_position']}]\n{item['chunk']}"
        for item in retrieved_items
    )
    
    # Format the last 3 turns (last 6 messages)
    history_text = ""
    if history:
        recent_history = history[-6:]  # Keep only the last 6 messages (3 exchanges)
        history_formatted = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_formatted.append(f"{role}: {msg['content']}")
        history_text = "Recent Chat History:\n" + "\n".join(history_formatted) + "\n\n"

    return (
        "Answer the question based ONLY on the following context.\n"
        "If the context doesn't contain enough information, "
        "say 'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context}\n\n"
        f"{history_text}"
        f"Question: {query}\n\n"
        "Answer:"
    )

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def generate(prompt):
    try:
        response = client.chat.completions.create(
            model="google/gemma-4-e4b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Simulated Response because local LLM is offline: {prompt[-50:]}]"

class RAGPipeline:
    def __init__(self, chunk_size=200, overlap=50, top_k=5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.chunks = []
        self.embeddings = []
        self.sources = []
        self.metadata = []
        self.history = []  # Memory storage

    def index(self, documents, source_names=None):
        all_chunks = []
        all_sources = []
        all_metadata = []
        for i, doc in enumerate(documents):
            doc_chunks = chunk_text(doc, self.chunk_size, self.overlap)
            all_chunks.extend(doc_chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(doc_chunks))
            all_metadata.extend([{"chunk_position": j, "source": name} for j in range(len(doc_chunks))])

        self.chunks = all_chunks
        self.sources = all_sources
        self.metadata = all_metadata
        self.embeddings = [embed(chunk) for chunk in all_chunks]
        return len(all_chunks)

    def query(self, question, top_k=None):
        k = top_k or self.top_k
        query_emb = embed(question)
        results = search(query_emb, self.embeddings, k)

        retrieved = []
        for idx, score in results:
            retrieved.append({
                "chunk": self.chunks[idx],
                "source": self.sources[idx],
                "metadata": self.metadata[idx],
                "score": score,
                "index": idx
            })

        prompt = build_rag_prompt(question, retrieved, self.history)
        answer = generate(prompt)
        
        # Save this turn to history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "retrieved": retrieved
        }

SAMPLE_DOCUMENTS = [
    """Acme Corp Product Overview.
    Acme Corp offers three product tiers: Starter, Professional, and Enterprise.
    The Starter plan includes basic features for individual users at $29 per month.
    The Professional plan adds team collaboration, advanced analytics, and priority
    support for $99 per month per user. The Enterprise plan includes everything in
    Professional plus custom integrations, dedicated account management, SSO,
    audit logs, and a 99.99% uptime SLA. Enterprise pricing is custom and starts
    at $500 per month for up to 50 users.""",

    """Acme Corp Refund Policy.
    All standard plan customers are eligible for a full refund within 30 days of purchase.
    Enterprise plan customers receive an extended 60-day refund window with pro-rated refunds
    calculated from the date of cancellation. Refunds are processed within 5-7 business days.
    No refunds are available after the refund window closes. Customers must submit refund requests.""",

    """Acme Corp Security Practices.
    Acme Corp maintains SOC 2 Type II compliance. All data is encrypted at rest using AES-256
    and in transit using TLS 1.3. Customer data is stored in isolated tenants within AWS us-east-1."""
]

if __name__ == "__main__":
    print("=" * 70)
    print("Initializing RAG Pipeline and Indexing Sample Documents...")
    print("=" * 70)
    
    rag = RAGPipeline(chunk_size=50, overlap=10, top_k=2)
    source_names = ["product-overview.md", "refund-policy.md", "security.md"]
    num_chunks = rag.index(SAMPLE_DOCUMENTS, source_names)
    print(f"Indexed {num_chunks} chunks successfully.\n")

    # Let's perform a multi-turn conversation
    chat_turns = [
        "What plans does Acme Corp offer?",
        "How much does the Professional plan cost?",
        "What features are included in it?",
        "what about enterprise?",
        "How is customer data encrypted?",
        "What is the refund policy for enterprise customers?"
    ]

    for i, user_query in enumerate(chat_turns):
        print("=" * 70)
        print(f"TURN {i+1}: Query = \"{user_query}\"")
        print("=" * 70)
        
        result = rag.query(user_query)
        
        print("\n--- Generated Prompt Preview (Showing History Section) ---")
        prompt_lines = result["prompt"].split("\n")
        # Find where history starts
        history_started = False
        history_lines = []
        for line in prompt_lines:
            if "Recent Chat History:" in line:
                history_started = True
            if history_started:
                if "Question:" in line:
                    break
                history_lines.append(line)
        if history_lines:
            print("\n".join(history_lines))
        else:
            print("  [No history present in prompt yet]")
            
        print("\n--- LLM Response ---")
        print(f"Assistant: {result['answer']}\n")
