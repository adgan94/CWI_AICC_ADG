import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name="office_faces")

# Get the first 5 entries to verify they exist
results = collection.get(limit=5)
print(f"IDs in Vector DB: {results['ids']}")
print(f"Metadata associated: {results['metadatas']}")