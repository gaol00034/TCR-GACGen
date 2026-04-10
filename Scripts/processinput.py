import torch
from transformers import T5Tokenizer
import pickle
import random

with open('./Data/GEXexamples.pkl', "rb") as nf:
    gexexamples = pickle.load(f)

pep_tokenizer = T5Tokenizer.from_pretrained('./dkarthikeyan1/tcrt5_pre_tcrdb')
def randompick(cellsubtype, gexdataset, k = 1):
    set = []
    for e in gexdataset:
        if e[-1] == cellsubtype:
            set.append(e)
    return random.sample(set, k)

def PEP_embed_T5(X: list):
    return pep_tokenizer(X, return_tensors='pt', padding=True)
def _prepare_pep(p, mhc):
    X = '[PMHC]'+ p + '[SEP]' + mhc + '[EOS]'
    X = PEP_embed_T5([X])
    return X
def embedding_with_given_matrix_cond(cond, matrix):
    return matrix[cond]
def _prepare_enc_cond(enc_cond):
    cond = torch.tensor(
        embedding_with_given_matrix_cond(enc_cond, onehot)
    )
    return cond
def preprocessing(pep, mhc, cellsubtype):
    data = {}

    data['pep'] = _prepare_pep(pep, mhc)

    pickedgex = randompick(cellsubtype, gexexamples)[0]

    data['gex'], enc_cond = pickedgex[1], (pickedgex[0], pickedgex[-1])

    data['enc_cond'] = _prepare_enc_cond(enc_cond)

    return data