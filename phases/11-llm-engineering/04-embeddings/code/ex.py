from embeddings import (
    SimpleEmbedder,
    VectorIndex,
    SemanticSearchEngine,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    hamming_distance,
    binarize,
    chunk_text,
    chunk_by_sentences,
    truncate_embedding,
    SAMPLE_DOCUMENTS,
    HFTransformerEmbedder
)

QUERIES = [
    "What is the refund policy for enterprise customers?",
    "What are the API rate limits?",
    "How is customer data encrypted?",
    "What happens if uptime falls below the SLA?",
    "How much does the Professional plan cost?"
]

SOURCE_NAMES = [
    "refund-policy.md",
    "product-overview.md",
    "security.md",
    "api-docs.md",
    "uptime-sla.md"
]                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                        
def run_excercise1():
                                                                                                                                 
    embedder = HFTransformerEmbedder("BAAI/bge-small-en-v1.5")        

    engine = SemanticSearchEngine(chunk_size=50, overlap=10 , embedder=embedder)
    engine.index_documents(SAMPLE_DOCUMENTS , SOURCE_NAMES)
    
    for q_idx , query in enumerate(QUERIES):
        cosine = engine.search(query , top_k=3 , metric="cosine")                                                                                            
        dot = engine.search(query , top_k=3 , metric="dot")                                                                                            
        euclidean = engine.search(query , top_k=3 , metric="euclidean")                                                                                            

        print("\n" + "="*100 + "\n")
        print("query : " , query)
        print("cosine : " , [r['index'] for r in cosine])
        print("dot : " , [r['index'] for r in dot])
        print("euclidean : " , [r['index'] for r in euclidean])
        print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    run_excercise1()
