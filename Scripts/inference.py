
import torch.optim
from seq2seq.tcrpep.train import *
from Model.gatedModel import condTransformer
import top_p_sampling_bidirection
import torch
import copy
from processinput import preprocessing


def load_checkpoint(filepath, model):
    state_dict = copy.deepcopy(model.state_dict())
    params = torch.load(filepath, map_location=torch.device('cpu'))
    # print()
    for key in params:
        if key in state_dict:
            state_dict[key] = params[key]
    return state_dict


def get_generated_len(seq: torch.Tensor):
    count = 0
    for i in range(seq.shape[0]):
        if seq[i] != 22:
            count += 1
        elif seq[i] == 21 or seq[i] == 0:
            continue
        else:
            count += 1
            break
    s = seq
    s[count:] = 0
    return s[:count - 1], count


def cut_generated_seq_with_tarlen(seq: str, len):
    return seq[:len]


def process_gen(generated_seq):
    generated_seq, generated_len = get_generated_len(generated_seq)
    generated_seq_str = onehot_to_str(generated_seq)
    return generated_seq_str


def reverse_concat(lr, rl):
    return lr + rl[::-1]

def reverse_concat_sets(lrs:list, rls:list):
    concat_l = []
    for i in lrs:
        for j in rls:
            concat_l.append(reverse_concat(i, j))
    return list(set(concat_l))



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = condTransformer()

model.load_state_dict(torch.load('./ckpt/tcrgacgen.pt'))

model.to(device)

file_path = './output/'

data = preprocess('KLGGALQAK', 'YFAMYQENVAQTDVDTLYIIYRDYTWAELAYTWY', 'Tpex')

model.eval()

with torch.no_grad():
    torch.cuda.empty_cache()

    pep = torch.tensor(data['pep']['input_ids']).to(device), torch.tensor(data['pep']['attention_mask']).to(device)

    gex = data['gex'].to(device)

    enc_cond = data['enc_cond'].to(device)

    iter_n = 100

    generated_cdr3bs_lrs = []
    generated_cdr3bs_rls = []

    generated_cdr3bs = []
    for j in range(iter_n):
        cdr_lr = top_p_sampling_bidirection(model, pep, gex, enc_cond, device, max_tokens=lrlen, top_p=0.6,
                                            direction='lr')
        cdr_rl = top_p_sampling_bidirection(model, pep, gex, enc_cond, device, max_tokens=rllen, top_p=0.6,
                                            direction='rl')

        generated_cdr3bs_lr = process_gen(cdr_lr)
        generated_cdr3bs_rl = process_gen(cdr_rl)

        generated_cdr3bs_lrs.append(generated_cdr3bs_lr)
        generated_cdr3bs_rls.append(generated_cdr3bs_rl)

    generated_cdr3bs_lrs, generated_cdr3bs_rls = list(set(generated_cdr3bs_lrs)), list(
        set(generated_cdr3bs_rls))
    generated_cdr3bs = reverse_concat_sets(generated_cdr3bs_lrs, generated_cdr3bs_rls)
    with open(file_path + 'gen.txt', mode="w") as file:
        data_to_write = "\n".join(generated_cdr3bs)
        file.write(data_to_write)
