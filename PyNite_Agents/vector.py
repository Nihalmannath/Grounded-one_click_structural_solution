from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import numpy as np

df_beams = pd.read_csv(r"Reference\beams.csv")
df_columns = pd.read_csv(r"Reference\columns.csv")
df_nodes = pd.read_csv(r"Reference\nodes.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    beam_lengths = []
    
    # Process beams data
    for i, row in df_beams.iterrows():
        node1_id = row['node1']
        node2_id = row['node2']
        
        # Find node coordinates
        node1_data = df_nodes[df_nodes['node_id'] == node1_id].iloc[0]
        node2_data = df_nodes[df_nodes['node_id'] == node2_id].iloc[0]
        
        # Calculate beam length (distance between nodes)
        length = np.sqrt((node2_data['x'] - node1_data['x'])**2 + 
                          (node2_data['y'] - node1_data['y'])**2 + 
                          (node2_data['z'] - node1_data['z'])**2)
        
        beam_lengths.append(length)
        
        document = Document(
            page_content = f"Beam {row['beam_id']} connects node {node1_id} to node {node2_id} with length {length:.2f}",
            metadata = {
                "type": "beam",
                "beam_id": row['beam_id'],
                "node1": node1_id,
                "node2": node2_id,
                "length": length
            },
            id = f"beam_{i}"
        )
        ids.append(f"beam_{i}")
        documents.append(document)
    
    # Process columns data
    for i, row in df_columns.iterrows():
        document = Document(
            page_content = f"Column {row['column_id']}",
            metadata = {
                "type": "column",
                "column_id": row['column_id']
            },
            id = f"column_{i}"
        )
        ids.append(f"column_{i}")
        documents.append(document)
    
    # Process nodes data
    for i, row in df_nodes.iterrows():
        document = Document(
            page_content = f"Node {row['node_id']} at coordinates ({row['x']}, {row['y']}, {row['z']})",
            metadata = {
                "type": "node",
                "node_id": row['node_id'],
                "x": row['x'],
                "y": row['y'],
                "z": row['z']
            },
            id = f"node_{i}"
        )
        ids.append(f"node_{i}")
        documents.append(document)
    
    print(f"Total beam lengths: {len(beam_lengths)}")
    print(f"Average beam length: {np.mean(beam_lengths):.2f}")

vector_store = Chroma(
    collection_name = "Structure_length",
    persist_directory = db_location,
    embedding_function = embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
