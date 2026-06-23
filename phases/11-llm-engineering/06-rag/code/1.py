from collections import Counter
import math
from openai import OpenAI
def chunk_text(text , chunk_size=200 , overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# binary bag of words
def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    sorted_vocab = sorted(vocab)
    return sorted_vocab, {word: i for i, word in enumerate(sorted_vocab)}

def binary_bag_of_words(text , vocab_list, word_to_index):
    vector = [0] * len(vocab_list)
    words = text.lower().split()
    for word in words:
        if word in word_to_index:
            vector[word_to_index[word]] = 1
    return vector

# def compute_tf(text , vocab):
#     words = text.lower().split()
#     count = Counter(words)
#     total = len(words)
#     if total == 0:
#         return [0.0] * len(vocab)
#     return [count.get(word , 0) / total for word in vocab]

# def compute_idf(documents , vocab):
#     n = len(documents)
#     idf = []
#     for word in vocab:
#         doc_count = sum(1 for doc in documents if word in doc.lower().split())
#         idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
#     return idf

# def tfidf_embed(text , vocab , idf):
#     tf = compute_tf(text , vocab)
#     return [t * i for t , i in zip(tf , idf)]

def cosine_similarity(a,b):
    dot_product = sum(x * y for x , y in zip(a , b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def search(query_embedding , stored_embeddings , top_k=5):
    scores = []
    for i , emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding , emb)
        scores.append((i ,sim))
    scores.sort(key=lambda x: x[1] , reverse=True)
    return scores[:top_k]

def build_rag_prompt(query , retrieved_items):
    context = "\n\n---\n\n".join(
        f"[Source {item['metadata']['source']}, Chunk {item['metadata']['chunk_position']}]\n{item['chunk']}"
        for item in retrieved_items
    )
    return(
        "Answer the question based ONLY on the following context.\n"
        "If the context doesn't contain enough information, "
        "say \"I don't have enough information to answer that.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

# actual llm call
from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:1234/v1",
    api_key = "lm-studio"
)

# def embed(text):
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=text
#     )
#     return response.data[0].embedding

def generate(prompt):
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

class RAGPipeline:
    def __init__(self , chunk_size=200 , overlap=50 , top_k=5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.word_to_index = {}
        self.sources = []
        self.metadata = []

    def index(self, documents , source_names=None):
        all_chunks = []
        all_sources = []
        all_metadata = []
        for i , doc in enumerate(documents):
            doc_chunks = chunk_text(doc , self.chunk_size , self.overlap)
            all_chunks.extend(doc_chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(doc_chunks))
            all_metadata.extend([{"chunk_position": j, "source": name} for j in range(len(doc_chunks))])

        self.chunks = all_chunks
        self.sources = all_sources
        self.metadata = all_metadata
        self.vocab, self.word_to_index = build_vocabulary(all_chunks)
        self.embeddings = [
            binary_bag_of_words(chunk, self.vocab, self.word_to_index)
            for chunk in all_chunks
        ]
        return len(all_chunks)
    
    def query(self , question , top_k = None):
        k = top_k or self.top_k
        query_emb = binary_bag_of_words(question, self.vocab, self.word_to_index)
        results = search(query_emb , self.embeddings , k)

        retrieved = []
        for idx , score in results:
            retrieved.append(
                {
                    "chunk" : self.chunks[idx],
                    "source" : self.sources[idx],
                    "metadata": self.metadata[idx],
                    "score" : score,
                    "index" : idx
                }
            )
        
        prompt = build_rag_prompt(question , retrieved)
        answer = generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "retrieved": retrieved
        }

SAMPLE_DOCUMENTS = [
    """Acme Corp Refund Policy.
    All standard plan customers are eligible for a full refund within 30 days of purchase.
    Enterprise plan customers receive an extended 60-day refund window with pro-rated refunds
    calculated from the date of cancellation. Refunds are processed within 5-7 business days
    and returned to the original payment method. No refunds are available after the refund
    window closes. Customers must submit refund requests through the support portal or by
    contacting their account manager directly. Annual subscriptions that are cancelled mid-term
    will receive a pro-rated credit for the remaining months.""",

    """Acme Corp Product Overview.
    Acme Corp offers three product tiers: Starter, Professional, and Enterprise.
    The Starter plan includes basic features for individual users at $29 per month.
    The Professional plan adds team collaboration, advanced analytics, and priority
    support for $99 per month per user. The Enterprise plan includes everything in
    Professional plus custom integrations, dedicated account management, SSO,
    audit logs, and a 99.99% uptime SLA. Enterprise pricing is custom and starts
    at $500 per month for up to 50 users. All plans include a 14-day free trial
    with no credit card required.""",

    """Acme Corp Security Practices.
    Acme Corp maintains SOC 2 Type II compliance and undergoes annual third-party
    security audits. All data is encrypted at rest using AES-256 and in transit
    using TLS 1.3. Customer data is stored in isolated tenants within AWS
    us-east-1 and eu-west-1 regions. Data residency can be configured per
    organization for Enterprise customers. Backups are performed every 6 hours
    with 30-day retention. Acme Corp does not sell or share customer data with
    third parties. Enterprise customers can request data deletion within 24 hours.
    Bug bounty program available through HackerOne.""",

    """Acme Corp API Documentation.
    The Acme API uses REST with JSON request and response bodies. Authentication
    is via Bearer tokens issued through OAuth 2.0. Rate limits are 100 requests
    per minute for Starter, 1000 for Professional, and 10000 for Enterprise.
    Rate limit headers are included in every response: X-RateLimit-Limit,
    X-RateLimit-Remaining, and X-RateLimit-Reset. Exceeding the rate limit
    returns HTTP 429 with a Retry-After header. The API supports pagination
    via cursor-based pagination using the next_cursor field. Webhooks are
    available for real-time event notifications on Professional and Enterprise
    plans. API versioning uses date-based versions in the URL path.""",

    """Acme Corp Uptime and Reliability.
    Acme Corp guarantees 99.9% uptime for Professional plans and 99.99% uptime
    for Enterprise plans. Uptime is calculated monthly excluding scheduled
    maintenance windows which are announced 72 hours in advance. If uptime
    falls below the guaranteed level, customers receive service credits:
    10% credit for each 0.1% below the SLA threshold, up to a maximum of
    30% of the monthly fee. Service credits must be requested within 30 days
    of the incident. Status page updates are posted at status.acme.com
    within 5 minutes of any detected incident. Post-incident reports are
    published within 48 hours for any outage exceeding 15 minutes."""
]


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Document Chunking")
    print("=" * 60)

    sample = SAMPLE_DOCUMENTS[0]
    chunks = chunk_text(sample, chunk_size=30, overlap=10)
    print(f"  Document length: {len(sample.split())} words")
    print(f"  Chunk size: 30 words, overlap: 10 words")
    print(f"  Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i}: ({len(chunk.split())} words)")
        print(f"    {chunk[:100]}...")

    print("\n" + "=" * 60)
    print("STEP 2: Binary Bag of Words Embedding")
    print("=" * 60)

    mini_docs = [
        "The cat sat on the mat",
        "The dog sat on the rug",
        "Machine learning is a branch of artificial intelligence"
    ]
    vocab, word_to_index = build_vocabulary(mini_docs)

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Sample words in vocabulary:")
    for word in vocab[:8]:
        print(f"    {word}")

    emb1 = binary_bag_of_words(mini_docs[0], vocab, word_to_index)
    emb2 = binary_bag_of_words(mini_docs[1], vocab, word_to_index)
    emb3 = binary_bag_of_words(mini_docs[2], vocab, word_to_index)

    print(f"\n  Embedding dimensions: {len(emb1)}")
    print(f"  Non-zero entries in 'cat sat on mat': {sum(1 for v in emb1 if v > 0)}")
    print(f"  Non-zero entries in 'dog sat on rug': {sum(1 for v in emb2 if v > 0)}")
    print(f"  Non-zero entries in 'machine learning': {sum(1 for v in emb3 if v > 0)}")

    print("\n" + "=" * 60)
    print("STEP 3: Cosine Similarity")
    print("=" * 60)

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)
    sim_23 = cosine_similarity(emb2, emb3)

    print(f"  'cat on mat' vs 'dog on rug':     {sim_12:.4f}  (similar structure)")
    print(f"  'cat on mat' vs 'machine learning': {sim_13:.4f}  (unrelated)")
    print(f"  'dog on rug' vs 'machine learning': {sim_23:.4f}  (unrelated)")
    print(f"\n  As expected: similar sentences score higher.")

    print("\n" + "=" * 60)
    print("STEP 4: Full RAG Pipeline")
    print("=" * 60)

    rag = RAGPipeline(chunk_size=50, overlap=10, top_k=3)
    source_names = [
        "refund-policy.md",
        "product-overview.md",
        "security.md",
        "api-docs.md",
        "uptime-sla.md"
    ]
    num_chunks = rag.index(SAMPLE_DOCUMENTS, source_names)
    print(f"  Indexed {len(SAMPLE_DOCUMENTS)} documents into {num_chunks} chunks")
    print(f"  Vocabulary size: {len(rag.vocab)} terms")

    queries = [
        "What is the refund policy for enterprise customers?",
        "What are the API rate limits?",
        "How is customer data encrypted?",
        "What happens if uptime falls below the SLA?",
        "How much does the Professional plan cost?"
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        result = rag.query(query, top_k=3)
        print(f"  Answer: {result['answer']}")
        print(f"  Retrieved {len(result['retrieved'])} chunks:")
        for r in result["retrieved"]:
            preview = r["chunk"][:80].replace("\n", " ")
            print(f"    [{r['source']}] score={r['score']:.4f} | {preview}...")

    print("\n" + "=" * 60)
    print("STEP 5: Chunk Size Comparison")
    print("=" * 60)

    test_query = "What is the refund policy for enterprise customers?"
    for chunk_size in [20, 50, 100, 200]:
        rag_test = RAGPipeline(chunk_size=chunk_size, overlap=max(5, chunk_size // 5))
        n = rag_test.index(SAMPLE_DOCUMENTS)
        result = rag_test.query(test_query, top_k=3)
        top_score = result["retrieved"][0]["score"] if result["retrieved"] else 0
        print(f"  chunk_size={chunk_size:>3d}: {n:>3d} chunks, "
              f"top_score={top_score:.4f}, "
              f"answer_len={len(result['answer'])}")

    print("\n" + "=" * 60)
    print("STEP 6: Prompt Inspection")
    print("=" * 60)

    result = rag.query("What encryption does Acme use?", top_k=2)
    prompt_lines = result["prompt"].split("\n")
    print(f"  Prompt length: {len(result['prompt'])} chars")
    print(f"  Prompt lines: {len(prompt_lines)}")
    print(f"\n  First 5 lines of generated prompt:")
    for line in prompt_lines[:5]:
        print(f"    {line}")
    print(f"  ...")
    print(f"  Last 3 lines of generated pro")
    for line in prompt_lines[-3:]:
        print(f"    {line}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  RAG pipeline: Query -> Embed -> Search -> Augment -> Generate")
    print(f"  Documents indexed: {len(SAMPLE_DOCUMENTS)}")
    print(f"  Total chunks: {num_chunks}")
    print(f"  Vocabulary size: {len(rag.vocab)}")
    print(f"  Embedding dimensions: {len(rag.vocab)}")
    print("  Similarity metric: cosine similarity")
    print("  Embedding method: Binary Bag of Words")
    print("\n  In production, replace Binary BoW with neural embeddings")
    print("  (text-embedding-3-small) and the simple generator with")
    print("  an actual LLM API call. The pipeline stays the same.")
