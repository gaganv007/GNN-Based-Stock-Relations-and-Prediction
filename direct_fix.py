import os
import pickle
import config

# Fix gpickle references to pkl
for fname in ['graph_construction.py']:
    text = open(fname).read()
    text = text.replace('nx.read_gpickle', 'pickle.load(open').replace('.gpickle', '.pkl')
    open(fname, 'w').write(text)

print("Fixed graph pickling in graph_construction.py")
