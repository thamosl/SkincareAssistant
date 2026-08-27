import sys
print(sys.version)
import torch
import transformers
import sentence_transformers
from langchain_community.embeddings import HuggingFaceEmbeddings

print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('sentence_transformers', sentence_transformers.__version__)
emb = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
print('embeddings init ok', type(emb).__name__)
