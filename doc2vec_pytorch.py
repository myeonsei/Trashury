# data format
# if document is embedded: [DOC_ID] [WORD] [WORD] [WORD] [WORD] ..\n
# otherwise: [WORD] [WORD] [WORD] [WORD] ..\n
# here coupang corpus doesn't need document embedding whereas store needs.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from collections import defaultdict
import numpy as np
import linecache

class SkipGramModel(nn.Module):
    def __init__(self, emb_size, doc_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.doc_size = doc_size
        self.emb_dimension = emb_dimension

        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        self.d_embeddings = nn.Embedding(self.doc_size, self.emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.d_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0) # why?
        
    def forward(self, pos_u, pos_v, neg_v, pos_d):
        
        emb_u = self.u_embeddings(pos_u) # Batches, Words, Dimension 
        emb_v = self.v_embeddings(pos_v) # Batches, Dimension 
        emb_neg_v = self.v_embeddings(neg_v) # Batches, Words, Dimension 
        
        if type(pos_d) is int: # 문서 벡터 임베딩 안 할 경우에는 이렇게 할 것 --
            input_ = torch.sum(emb_u, dim=1) / emb_u.size(1) # Batches, Dimension
            pos = (emb_v.unsqueeze(1) @ input_.unsqueeze(-1)).squeeze() # Batches
            neg = torch.sum(torch.exp(torch.clamp((emb_neg_v @ input_.unsqueeze(-1)).squeeze(), max=50)), dim=1) # Batches
            
            pos = torch.clamp(pos, max=50)
            loss = -torch.mean(pos - torch.log(torch.exp(pos) + neg))
        
        else:
            emb_d = self.d_embeddings(pos_d) # Batches, Dimension 
            input_ = (torch.sum(emb_u, dim=1) + emb_d) / (emb_u.size(1) + 1) # Batches, Dimension
            pos = (emb_v.unsqueeze(1) @ input_.unsqueeze(-1)).squeeze() # Batches
            neg = torch.sum(torch.exp(torch.clamp((emb_neg_v @ input_.unsqueeze(-1)).squeeze(), max=50)), dim=1) # Batches
            
            pos = torch.clamp(pos, max=50)
            loss = -torch.mean(pos - torch.log(torch.exp(pos) + neg))
            
        return loss
    
    def save(self, path):
        torch.save(model.state_dict(), path)
        
def build_vocab(store_corpus, coupang_corpus, batch_size, min_frq = 30):
    counter = defaultdict(lambda: 0)
    doc_map = {}
    store_length = 0
    coupang_length = 0
    word_count = 0
    
    with open(store_corpus, 'r') as f:
        while True:
            tmp = f.readline()
            if not tmp: break
            store_length += 1
            
            tmp = tmp.strip().split(' ') # 무조건 11개 이상 단어 남도록 할 것...
            word_count += (len(tmp) - 1)
            doc_map[store_length-1] = tmp[0]
            for word in tmp[1:]:
                counter[word] += 1
    
    store_batch = (word_count + batch_size - 1) // batch_size
   
    word_count = 0
    
    with open(coupang_corpus, 'r') as f:
        while True:
            tmp = f.readline()
            if not tmp: break
            coupang_length += 1
            
            tmp = tmp.strip().split(' ')
            word_count += len(tmp)
            for word in tmp:
                counter[word] += 1
                
    coupang_batch = (word_count + batch_size - 1) // batch_size
             
    counter = sorted([(key, count) for key, count in counter.items() if count >= min_frq], key=lambda x: -x[1])
        
    return store_length, coupang_length, store_batch, coupang_batch, dict(counter), dict([(y, x) for x, y in enumerate([k for k, c in counter])]), doc_map
    
batch_size = 128
store_length, coupang_length, store_batch, coupang_batch, count_dict, word_map, doc_map = build_vocab('doc2vec_corpus/store_corpus.txt', 'doc2vec_corpus/coupang_corpus.txt', batch_size, min_frq = 20)
# reverse_doc_map = dict([(y, x) for x, y in doc_map.items()])
word_count = len(count_dict)
prob = [x ** 3/4 for _, x in sorted(word_map.items(), key=lambda x: x[1])]
prob = [x/sum(prob) for x in prob]
unk = len(word_map)
doc_map2 = dict([(y, x) for x, y in enumerate(set([x[1] for x in doc_map.items()]))])
doc_count = len(doc_map2)

window_size = 5
epoches = 20
model = SkipGramModel(word_count+1, doc_count, 100)
model.cuda()

store_corpus = 'doc2vec_corpus/store_corpus.txt'
coupang_corpus = 'doc2vec_corpus/coupang_corpus.txt'
optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (coupang_batch + store_batch) * epoches)

for e in range(epoches):
    print(f'Epoch {e+1}')
    count = 0 
    learning_loss = 0.0
    
    c_line_indices = list(range(coupang_length)) # 일단 문서 빠진 것
    np.random.shuffle(c_line_indices)
    
    s_line_indices = list(range(store_length)) # 일단 문서 빠진 것
    np.random.shuffle(s_line_indices)
   
    batch_order = ['C'] * coupang_batch + ['S'] * store_batch
    np.random.shuffle(batch_order)
    
    c_pos_u = []; c_pos_v = []
    s_pos_u = []; s_pos_v = []; s_pos_d = []
    
    for b in batch_order:
        if b == 'C':
            while len(c_pos_v) < batch_size:
                if len(c_line_indices) == 0: break
                sentences = linecache.getline(coupang_corpus, c_line_indices[0] + 1).strip().split(' ')
                c_line_indices = c_line_indices[1:]

                for j in range(len(sentences)):
                    if j - window_size >= 0 and j + window_size < len(sentences):
                        c_pos_u.append(sentences[(j-window_size):j] + sentences[(j+1):(j+window_size+1)])
                    elif j - window_size < 0:
                        c_pos_u.append(sentences[0:j] + sentences[(j+1):(2*window_size+1)])
                    else:
                        c_pos_u.append(sentences[(len(sentences)-2*window_size-1):j] + sentences[(j+1):len(sentences)])

                    c_pos_u[-1] = [word_map[x] if x in word_map.keys() else unk for x in c_pos_u[-1]]
                    c_pos_v.append(word_map[sentences[j]] if sentences[j] in word_map.keys() else unk)
            # 왠지 unk가 문제일 거 같은데 . ..
            tmp_u = torch.tensor(c_pos_u[:batch_size], dtype=torch.long).cuda()
            tmp_v = torch.tensor(c_pos_v[:batch_size], dtype=torch.long).cuda()
            c_pos_u = c_pos_u[batch_size:]; c_pos_v = c_pos_v[batch_size:]
            tmp_d = 0
            
        else:
            while len(s_pos_v) < batch_size:
                if len(s_line_indices) == 0: break
                sentences = linecache.getline(store_corpus, s_line_indices[0] + 1).strip().split(' ')
                s_line_indices = s_line_indices[1:]
                d = sentences[0]; sentences = sentences[1:]
                
                for j in range(len(sentences)):
                    if j - window_size >= 0 and j + window_size < len(sentences):
                        s_pos_u.append(sentences[(j-window_size):j] + sentences[(j+1):(j+window_size+1)])
                    elif j - window_size < 0:
                        s_pos_u.append(sentences[0:j] + sentences[(j+1):(2*window_size+1)])
                    else:
                        s_pos_u.append(sentences[(len(sentences)-2*window_size-1):j] + sentences[(j+1):len(sentences)])

                    s_pos_u[-1] = [word_map[x] if x in word_map.keys() else unk for x in s_pos_u[-1]]
                    s_pos_v.append(word_map[sentences[j]] if sentences[j] in word_map.keys() else unk)
                    s_pos_d.append(doc_map2[d])
    
            tmp_d = torch.tensor(s_pos_d[:batch_size], dtype=torch.long).cuda()
            tmp_u = torch.tensor(s_pos_u[:batch_size], dtype=torch.long).cuda()
            tmp_v = torch.tensor(s_pos_v[:batch_size], dtype=torch.long).cuda()
            
            s_pos_u = s_pos_u[batch_size:]
            s_pos_v = s_pos_v[batch_size:]
            s_pos_d = s_pos_d[batch_size:]

        neg_v = np.random.choice(np.arange(word_count), tmp_v.size(0)*5, p=prob).reshape(tmp_v.size(0), -1)
        neg_v = torch.tensor(neg_v, dtype=torch.long).cuda()

        scheduler.step()
        optimizer.zero_grad()
        loss = model(tmp_u, tmp_v, neg_v, tmp_d)
        loss.backward()
        learning_loss += loss
        optimizer.step()

        count += 1
        if count % 10000 == 0:
            print(f'{count}th Batch Processed / Average Loss So Far: {learning_loss / count}')
#             print((model.u_embeddings.weight).abs().max())

        learning_loss += loss.item()
        
    print(f'Average Loss: {learning_loss / (coupang_batch + store_batch)}')
