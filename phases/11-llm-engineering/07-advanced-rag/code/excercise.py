from main import (
    SAMPLE_DOCUMENTS , chunk_text , build_vocabulary , compute_tf , compute_idf , tfidf_embed , cosine_similarity , vector_search , BM25 , reciprocal_rank_fusion , hybrid_search
)

all_chunks = []
chunk_sources = []
chunk_metadata = []
categories = ["billing", "product", "security", "api", "billing", "product"]
source_names = ["refund", "product", "security", "api", "earnings", "uptime"]

for i , doc in enumerate(SAMPLE_DOCUMENTS):
    chunks = chunk_text(doc , chunk_size=50 , overlap=10)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_sources.append(source_names[i])
        # metadata dict
        chunk_metadata.append({"source": source_names[i], "category": categories[i]})
# print(chunk_metadata)
# queries = [
#     "What is the refund policy for enterprise customers?",
#     "What was money made last quarter?",
#     "Data encryption process?",
#     "What are the API rate limits for enterprise?",
#     "What happens if uptime falls below SLA?"
# ]

# bm25

# bm25 = BM25()
# bm25.index(all_chunks)

# vector search
stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
              "why", "when", "where", "do", "does", "for", "of", "in", "to",
              "and", "or", "on", "at", "by", "it", "its", "this", "that",
              "with", "from", "be", "has", "have", "had", "not", "but"}
vocab = [w for w in build_vocabulary(all_chunks) if w not in stop_words]
idf = compute_idf(all_chunks, vocab)
embeddings = [tfidf_embed(c, vocab, idf) for c in all_chunks]

def filtered_vector_search(query_embedding, stored_embeddings, metadata, target_category, top_k=5):  
    scores = []                                                                                      
    for i, emb in enumerate(stored_embeddings):                                                      
        category = metadata[i]["category"]                   
        if category == target_category:
            sim = cosine_similarity(query_embedding, emb)                                                
            scores.append((i, sim))                                                                      
                                                                                                        
    scores.sort(key=lambda x: x[1], reverse=True)                                                    
    return scores[:top_k]

# for i , query in enumerate(queries):
#     print(f"Query: {query}")
    
    # bm25_res = bm25.search(query, top_k=2)
    # if bm25_res:
    #     idx, score = bm25_res[0]
    #     print(f"BM25 (score: {score:.4f}): {all_chunks[idx]}")
    
    # query_embedding = tfidf_embed(query, vocab, idf)

    # vector_res = filtered_vector_search(query_embedding, embeddings,chunk_metadata , 'api', top_k=2)
    # if vector_res:
    #     idx, score = vector_res[0]
    #     print(f"Vector (score: {score:.4f}): {all_chunks[idx]}")
        
    # hybrid_res = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=2)
    # if hybrid_res:
    #     idx, score = hybrid_res[0]
    #     print(f"Hybrid (score: {score:.4f}): {all_chunks[idx]}")
        
    # print("-" * 50)

# ex-3 hyde vs direct query
from openai import OpenAI

client = OpenAI(
    base_url='http://127.0.0.1:1234/v1',
    api_key='lm-studio'
)

def generate(prompt):
    response = client.chat.completions.create(                                               
        model="google/gemma-4-e4b",                                                          
        messages=[{"role": "user", "content": prompt}],                                      
        temperature=0                                                                        
    )                                                                                        
    return response.choices[0].message.content

def print_retrieved_results(results, chunks, sources):                                           
    for rank, (idx, score) in enumerate(results):                                                
        source_name = sources[idx]                                                               
        text_preview = chunks[idx][:120].replace("\n", " ")                                      
        print(f"    #{rank+1} [{source_name}] (score: {score:.4f}) | {text_preview}...")

def hyde_search_llm(query, vector_embeddings, vocab, idf, top_k):
    prompt = f"""Generate a detailed and comprehensive paragraph answering the following question.
    Write it in the style of a professional documentation page.

    Question: {query}

    Instructions:
    - Provide a clear, factual answer.
    - Include specific details and relevant information.
    - Write in a formal, technical tone.
    - Make it sound like an expert wrote this.

    Paragraph:"""

    hypothesis = generate(prompt)
    hyp_emb = tfidf_embed(hypothesis , vocab , idf)

    results = vector_search(hyp_emb , vector_embeddings , top_k=top_k)

    return results , hypothesis


queries = [
     "How much revenue did the company make?",

]

for i , query in enumerate(queries):
    print(f"\nQuery: '{query}'")                                                                 
    print("=" * 60)                                                                              
                                                                                                    
    # 1. Direct Search                                                                           
    print("Direct Query Search Results:")                                                        
    query_emb = tfidf_embed(query, vocab, idf)                                                   
    direct_results = vector_search(query_emb, embeddings, top_k=3)                               
    print_retrieved_results(direct_results, all_chunks, chunk_sources)                           
                                                                                                    
    print("-" * 60)                                                                              
                                                                                                    
    # 2. HyDE Search                                                                             
    print("HyDE Search Results:")                                                                
    hyde_results, hypothesis = hyde_search_llm(query, embeddings, vocab, idf, top_k=3)           
    print(f"  [Hypothesis Generated]:\n  \"{hypothesis[:200]}...\"\n")                           
    print_retrieved_results(hyde_results, all_chunks, chunk_sources)

