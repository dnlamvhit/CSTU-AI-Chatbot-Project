# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:28:32 2024

@author: lamngocdao
"""
from PyPDF2 import PdfReader
from time import sleep

def update_kb_openai():
    # Using OpenAI embedding model
    embedding_model = "text-embedding-ada-002"
    
    # Create a reader object for the knowledge base file
    reader = PdfReader(file_name)
    page_len = len(reader.pages)
    # Extract text from each page and concatenate
    doc = []
    paragraph = ""
    for i in range(page_len):
        page_text = reader.pages[i].extract_text().splitlines()
        for line in page_text:
            if line.strip() == "":
                if paragraph:
                    doc.append(paragraph)
                    paragraph = ""
            else:
                paragraph += " " + line.strip()
    if paragraph:  # add the last paragraph if it's not empty
        doc.append(paragraph)
    
    # Set chunking and embedding parameters
    chunk_size = 1
    stride = 1
    cstu_id = "cstu-kb"
      
    # Iterate over the document in chunks and upsert embeddings to index
    count = 0 + pincone_index.describe_index_stats()['total_vector_count']
    result=str(count)
    i_begin=0; # beginning of the chunk
    while i_begin <= (len(doc) - chunk_size):
        i_end = min(len(doc), i_begin + chunk_size)
        doc_chunk = doc[i_begin:i_end]
        texts = ""
        for x in doc_chunk:
            texts += x
        print("The", count+1, "doc chunk text:", texts)
        print("==========================================================")
    
        try: res = openai.Embedding.create(input=texts, engine=embedding_model)
        except:
                done = False
                while not done:
                    sleep(6)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embedding_model)
                        done = True
                    except:
                        pass
        embedding = res['data'][0]['embedding']
    
        # Prepare metadata
        metadata = {"cstu_id": cstu_id + '_' + str(count), "text": texts}
        count += 1
    
        # Upsert to Pinecone and corresponding namespace
        pincone_index.upsert(vectors=[(metadata["cstu_id"], embedding, metadata)], namespace="cstu")
        i_begin += stride
    result = result + ' - ' + str(count-1)
    return result

def update_kb_cstu():
    # Generating and using CSTU embedding model
    embedding_model = load('CSTU-embedding-model.mdl')
    
    # Create a reader object for the knowledge base file
    reader = PdfReader(file_name)
    page_len = len(reader.pages)
    # Extract text from each page and concatenate to the CSTU scorpus (doc)
    doc = []
    paragraph = ""
    for i in range(page_len):
        page_text = reader.pages[i].extract_text().splitlines()
        for line in page_text:
            if line.strip() == "":
                if paragraph:
                    doc.append(paragraph)
                    paragraph = ""
            else:
                paragraph += " " + line.strip()
    if paragraph:  # add the last paragraph if it's not empty
        doc.append(paragraph)

    # Train Word2Vec model on entire corpus
    tokens = [nltk.word_tokenize(text) for text in doc]
    model = Word2Vec(tokens, vector_size=1536, min_count=1)
    model.save("CSTU-embedding-model.mdl")
    
    # Set chunking and embedding parameters
    chunk_size = 1
    stride = 1
    cstu_id = "cstu-kb"
      
    # Iterate over the document in chunks and upsert embeddings to index
    count = 0 + pincone_index.describe_index_stats()['total_vector_count']
    result=str(count)
    i_begin=0; # beginning of the chunk
    while i_begin <= (len(doc) - chunk_size):
        i_end = min(len(doc), i_begin + chunk_size)
        doc_chunk = doc[i_begin:i_end]
        texts = ""
        for x in doc_chunk:
            texts += x
        print("The", count+1, "doc chunk text:", texts)
        print("==========================================================")
    
        try: 
            tokens = nltk.word_tokenize(texts)
            word_vectors = [model.wv[token] for token in tokens if token in model.wv]
            if not word_vectors: # If no valid word vectors are found, return a vector of zeros
                return np.zeros(model.vector_size)
            embedding = np.mean(word_vectors, axis=0)            
        except Exception as e:
            st.write(e)
    
        # Prepare metadata
        metadata = {"cstu_id": cstu_id + '_' + str(count), "text": texts}
        count += 1
    
        # Upsert to Pinecone and corresponding namespace
        pincone_index.upsert(vectors=[(metadata["cstu_id"], embedding, metadata)], namespace="cstu")
        i_begin += stride
    result = result + ' - ' + str(count-1)
    return result
    
