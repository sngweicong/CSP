import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
from Layers import EncoderLayer #,DecoderLayer
from Sublayers import Norm
from Embed import PositionalEncoder
import matplotlib.pyplot as plt
import getInput as getInput
import time
import sys

#vertex_arr = np.load("../tsfm/vertex_arr_CHON.npy", allow_pickle=True) #1843
#mol_adj_arr = np.load("../tsfm/mol_adj_CHON.npy", allow_pickle=True)
#msp_arr = np.load("../tsfm/msp_arr_CHON.npy", allow_pickle=True)

vertex_arr = np.load("../tsfm/vertex_arr_test.npy", allow_pickle=True) #1843
mol_adj_arr = np.load("../tsfm/mol_adj_arr_test.npy", allow_pickle=True)
msp_arr = np.load("../tsfm/msp_arr_sort_per.npy", allow_pickle=True)

msp_len = 800
k = 20
atom_type = 2
padding_idx = 799  # must > 0 and < msp_len
dropout = 0.2
batch_size = 1
atom_mass = [12, 1, 16, 14]  # [12,1,16]

dict_atom = dict([ (atom_mass[i],i) for i in range(len(atom_mass))])
embed_idx_grid = torch.LongTensor([range(len(atom_mass)**2)]).reshape(len(atom_mass),len(atom_mass))
for i in range(1,len(atom_mass)):
    for j in range(i):
        embed_idx_grid[i,j] = embed_idx_grid[j,i]
#print(embed_idx_grid)

atomic_number = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]  # [C, H, O, N}
bond_number = [4, 1, 2]  # [C, H, O]
default_valence = [[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
MSP_THRESHOLD = 100
n_top_msp = 16

edge_num = 78  # 3#29*15#78 #3
d_model = 256
max_atoms = 13  # 3
max_len11 = edge_num + 16
max_len12 = 2 * edge_num + k

def get_atom_atom_embedding_idx(atom_mass_1,atom_mass_2):
    return embed_idx_grid[dict_atom[atom_mass_1], dict_atom[atom_mass_2]]

#for testing
#print(get_atom_atom_embedding_idx(12,14))
#print(get_atom_atom_embedding_idx(16,16))

def getEdgeIdx(pos1, pos2=None):  # not contain H
    edge_idx = 0
    for jj in range(pos1):
        edge_idx += (max_atoms - jj - 1)
    if pos2 == None:
        return edge_idx - 1
    else:
        edge_idx += pos2 - pos1
    return edge_idx - 1

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)  # .unsqueeze(-2)

# transformer with edge non-pos
class Classify11(nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.debug_explode_count = 0
        self.N = 1
        self.padding_idx = padding_idx
        self.msp_embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.edges_embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len11, 4)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, msp, edges):
        #firstmask = get_pad_mask11(src).bool().cuda()  # [batch, edge_num,4]
        #mask some modality
        #msp[:] = self.padding_idx
        #edges[:] = self.padding_idx
        #end mask
        temp_concat = torch.cat((msp,edges),1)
        mask = get_pad_mask(temp_concat, self.padding_idx).view(temp_concat.size(0), -1).unsqueeze(-1)
        #print(mask)
        output_msp = self.msp_embedding(msp)  # [batch, k, d_model=512]
        output_edges = self.edges_embedding(edges)
        output = torch.cat((output_msp,output_edges),1)
        #print('1', output.shape)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        #print('2', output.shape)
        output = self.ll1(output)  # [batch, max_len2, edge_num]
        #print('3', output.shape)
        #output = self.dropout1(output)
        #
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len2]
        #print('4', output.shape)
        output = self.ll2(output)  # [batch, edge_num, 4]
        #print('5', output.shape)
        '''
        1 torch.Size([1, 94, 256])
        2 torch.Size([1, 94, 256])
        3 torch.Size([1, 94, 78])
        4 torch.Size([1, 78, 94])
        5 torch.Size([1, 78, 4])
        '''
        #output = output.masked_fill(firstmask,-1e9)
        output2 = F.log_softmax(output, dim=-1)
        if torch.sum(output2) < -5000 and self.debug_explode_count < 20:
            print("EXPLODED MODEL", self.debug_explode_count)
            print(torch.sum(output2))
            print(output2)
            self.debug_explode_count += 1
            print(output)
        return output2  # [batch, edge_num=3, bond=4]

def getInputLite1Edge(vertex, msp):
    edges = torch.LongTensor([[padding_idx] * edge_num] * len(vertex))
    top16msp = torch.LongTensor([[padding_idx] * 16] * len(vertex))
    for b in range(len(vertex)):
        idx = 0
        for i in range(max_atoms):
            for ii in range(i + 1, max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    #print(i,ii,idx1)
                    #print("first",src[b, idx1 * 15: idx1 * 15 + 15])
                    #OLD edges[b, idx] = atom_mass[int(vertex[b][i])] + atom_mass[int(vertex[b][ii])]
                    edges[b, idx] = get_atom_atom_embedding_idx(atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])])
                idx += 1
        #MSP part
        top16_unfiltered = (-msp[b]).argsort()[:16]
        more_than_10_bool = (msp[b]>= MSP_THRESHOLD)
        end_idx = min(15,np.sum(more_than_10_bool))
        top16msp[b,:end_idx] = torch.Tensor(top16_unfiltered[:end_idx])
    #print(edges.shape) # 1 156
    #print(top16msp.shape) # 1 16 
    return top16msp, edges

def getLabel(mol_arr, vertex):
    # label = torch.zeros((len(mol_arr),edge_num),dtype=torch.long) #[batch, edge_num]
    label = torch.LongTensor([[padding_idx] * edge_num] * len(mol_arr))
    for b in range(len(mol_arr)):  # batch
        idx = 0
        for i in range(len(mol_arr[b])):
            for j in range(i + 1, len(mol_arr[b])):
                if i < len(vertex[b]) and j < len(vertex[b]):
                    label[b, idx] = mol_arr[b][i][j]
                idx += 1
    return label
def accuracy(preds_bond,label_graph,vertex):
    bs = len(label_graph)
    preds_graph = torch.zeros((bs,max_atoms,max_atoms)) #batch, max_atom=3, max_atom=3
    accs = []
    for b in range(bs):
        idx = 0
        acc = 0
        count = 0
        length = len(vertex[b])
        for i in range(max_atoms):
            for j in range(i+1, max_atoms):
                preds_graph[b,i,j] = preds_bond[b][idx]
                preds_graph[b, j, i] = preds_graph[b,i,j]
                idx +=1
                if i < length and j < length and [i,j] not in [[0,1],[0,2]]:
                    count += 1
                    if preds_graph[b,i,j]==label_graph[b,i,j]:acc+=1
        accs.append(round(acc/(count+np.finfo(np.float32).eps),4))
    return accs, preds_graph

def isValid(ori_bonds,pred_bond,vertex):
    bonds = ori_bonds + [pred_bond]
    preds_graph = torch.zeros((max_atoms, max_atoms))  # batch, max_atom=3, max_atom=3
    idx = 0
    for i in range(max_atoms):
        for j in range(i + 1, max_atoms):
            preds_graph[i, j] = bonds[idx]
            preds_graph[j, i] = preds_graph[i, j]
            idx += 1
            if idx >= len(bonds): break
        if idx >= len(bonds): break
    if i >= len(vertex) or j>= len(vertex): return 0
    sum_row = preds_graph[i].sum()
    sum_col = preds_graph[:,j].sum()
    max_row = bond_number[int(vertex[i])]
    max_col = bond_number[int(vertex[j])]
    if (max_row-sum_row) >= 1 and (max_col-sum_col) >= 1: return 1
    if (max_row - sum_row) >= 2 and (max_col - sum_col) >= 2: return 2
    if (max_row - sum_row) >= 3 and (max_col - sum_col) >= 3: return 3
    if (max_row - sum_row) >= 4 and (max_col - sum_col) >= 4: return 3
    else: return -1
# transformer with edge
def train11(model, epoch, num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    not_nan = True
    prev_loss = 0
    prev_pred = 0
    prev_loss_2 = 0
    prev_pred_2 = 0

    for batch, i in enumerate(range(0, len(num), batch_size)):
        #print("batch,i", batch, i)
        seq_len = min(batch_size, len(num) - i)
        #print('seq_len', seq_len)
        #print('vertex_data', vertex_data[i:i + seq_len])
        #print('msp_arr', msp_arr_data[i:i + seq_len])
        msp, edges = getInputLite1Edge(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        msp = msp.cuda()
        edges = edges.cuda()
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len]).cuda()
        #print('labels',labels)
        labels_graph = mol_adj_data[i:i + seq_len]
        #print('labels_graph',labels_graph)
        preds = model(msp,edges)  # batch, edge_num, 4
        #print('Preds',preds)
        preds_bond = torch.argmax(preds, dim=-1)  # batch edge_num
        #print(preds_bond)

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        if torch.sum(torch.isnan(loss)) > 0 and not_nan:
            not_nan = False
            print("Exploded")
            print(preds.view(-1, 4))
            print(loss)
            print(prev_pred)
            print(prev_loss)
            print(prev_pred_2)
            print(prev_loss_2)
            continue
        
        prev_loss_2 = prev_loss
        prev_pred_2 = prev_pred
        prev_loss = loss
        prev_pred = preds.view(-1,4)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        #if (epoch - 1) % 10 == 1 and i == 0:
            #print(labels_graph[0])
            #print(preds)
            #print(preds_bond)
            #print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss / len(num), 4))
    print("train mean_acc: ", round(sum(accs) / len(accs), 4))
    train_acc_list.append(round(sum(accs) / len(accs), 4))
    tran_loss_list.append(round(total_loss / len(num), 4))
    return sum(accs) / len(accs)
def evaluate11(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            #print("Eval", i)
            start_time = time.time()
            seq_len = min(batch_size, len(num) - i)
            msp, edges = getInputLite1Edge(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            msp = msp.cuda()
            edges = edges.cuda()
            #print("1. getinput11 time", time.time()-start_time)
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len]).cuda()
            labels_graph = mol_adj_data[i:i + seq_len]
            #print(labels_graph)
            #print(labels_graph.shape)
            preds = model(msp,edges)  # batch, 3, 4
            #print("2. model(src)", time.time()-start_time)
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            #print(vertex_data[i])
            #atom_lists = getInput.find_permutation(vertex_data[i])
            #print(atom_lists)
            #print("3. getInput.find_permu", time.time()-start_time)
            #losses = []
            #for al in atom_lists:
                #new_E = getInput.getGraph(labels_graph[0], al)
                #labels = getLabel([new_E], vertex_data[i:i + seq_len])
                #loss = criterion(preds.view(-1, 4), labels.view(-1))
                #losses.append(loss)
            #print("4. al in atom_lists time", time.time()-start_time)
            #loss = min(losses)
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            #if (epoch - 1) % 50 == 0 and i == 0:
                #print(labels_graph[0])
                #print(preds_graph[0])
        print("valid mean_loss: ", round(total_loss / len(num), 4))
        print("valid mean_accs: ", round(sum(accs) / len(accs), 4))
        valid_acc_list.append(round(sum(accs) / len(accs), 4))
        valid_loss_list.append(round(total_loss / len(num), 4))
def test11(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            msp, edges = getInputLite1Edge(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            msp = msp.cuda()
            edges = edges.cuda()
            #print('----------------')
            #print(vertex_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len]).cuda()
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(msp, edges)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            #print(labels_graph[0])
            #print(preds_graph[0])
        print("test mean_loss: ", round(total_loss / len(num), 4))
        print("test mean_accs: ", round(sum(accs) / len(accs), 4))
        test_acc_list.append(round(sum(accs) / len(accs), 4))
        test_loss_list.append(round(total_loss / len(num), 4))

def plot_result(epoch):
    x1 = range(0,epoch)
    x2 = range(0,epoch)
    y1 = train_acc_list
    y2 = tran_loss_list
    y3 = valid_acc_list
    y4 = valid_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '-', label="Train_Accuracy")
    plt.plot(x1, y3, '-', label="Valid_Accuracy")
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '-', label="Train_Loss")
    plt.plot(x2, y4, '-', label="Valid_Loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()
    
def train_transformer(epoch, num):
    for i in range(1, 1 + epoch):
        cur_time = time.time()
        train11(model, i, num)
        print("TrainTime",time.time() - cur_time)
        cur_time = time.time()
        evaluate11(model, i, range(1500, 1700))
        print("EvalTime",time.time() - cur_time)
        cur_time = time.time()
        test11(model, i, range(1700, 1800))
        print("TestTime",time.time() - cur_time)
        #print(model)
        norm_list = []
        for p in model.parameters():
            if p.grad is None:
                continue
            norm_list.append(p.grad.data.norm(2).item())
        print(norm_list)
    torch.save(model.state_dict(),'model_type11.pkl')
    plot_result(epoch)

model = Classify11(padding_idx)
torch.cuda.set_device(int(sys.argv[1]))
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.NLLLoss(ignore_index=padding_idx)  # CrossEntropyLoss()

train_acc_list, tran_loss_list, valid_acc_list, valid_loss_list, test_acc_list, test_loss_list = [],[],[],[], [], []
train_transformer(500,num=range(1500))
#train11(model, 1, range(5))

# #Testing
#model.load_state_dict(torch.load('type11_1epoch.pkl'))
#evaluate11(model,1,range(1339,1340))
