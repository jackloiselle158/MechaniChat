import chromadb
from tqdm import tqdm
import ollama

client = chromadb.PersistentClient(path="./mydb/")
collection = client.get_or_create_collection(name="docs")
documents = []
with open("nhtsa_complaints.txt", "r", encoding="utf-8") as f:
    for line in f:
        documents.append(line.strip())
print(len(documents))
#========================== Comment this out later ==========================
for i, doc in tqdm(enumerate(documents)):
    response = ollama.embed(model="nomic-embed-text", input=doc)
    embedding = response["embeddings"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding] if not isinstance(embedding[0], list) else embedding,
        documents=[doc]
    )  
#=============================================================================