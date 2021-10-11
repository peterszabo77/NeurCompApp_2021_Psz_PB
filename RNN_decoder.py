# https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil

RANDOMSEED = 1337

EMBEDDING_SIZE = 50
GRU_HIDDEN_SIZE = 400
GRU_LAYERS = 1
BIDIRECTIONAL_GRU = False
FC_HIDDEN_SIZE = 1000
REGULARIZATION = 'L2'
D_a = 32

N_EPOCHS = 200
BATCHSIZE = 100
LEARNING_RATE = 0.0001

DATADIR = 'data'
DATAFILES = {
'D_A_I': 'D_A_I.csv',
'D_A_II': 'D_A_II.csv',
'D_A_III': 'D_A_III.csv',
'D_E_I': 'D_E_I.csv',
'D_E_II': 'D_E_II.csv',
'D_E_III': 'D_E_III.csv'
}
DATASET = 'D_E_I'

OUTDIR = 'result'
SAVEMODEL = True

# DATASET
class MyDataset():
	def __init__(self,device):
		self.device = device
		filepath = os.path.join('.', DATADIR, DATAFILES[DATASET])
		alldata = pd.read_csv(filepath)
		self.N_allrecords = alldata.shape[0]
		self.sound_IDs = list(alldata['sound_ID'].sort_values().unique())
		self.n_classes = len(self.sound_IDs)
		print(self.n_classes,'sound classes')
		self.class_counts = alldata['sound_ID'].value_counts().to_dict()
		self.class_frequencies = alldata['sound_ID'].value_counts(normalize=True)
		counts = [self.class_counts[sound_ID] for sound_ID in self.sound_IDs]
		self.class_weights = 1.0 / torch.tensor(counts, dtype=torch.float32)
		self.cell_IDs = np.unique(alldata.drop(['role','sound_ID'],axis=1).to_numpy())
		self.cell_IDs = self.cell_IDs[self.cell_IDs>0]
		print('number of cells:', len(self.cell_IDs))
		self.df_train = alldata[alldata['role']=='train'].drop('role',axis=1)
		self.df_validation = alldata[alldata['role']=='val'].drop('role',axis=1)
		self.df_test = alldata[alldata['role']=='test'].drop('role',axis=1)
		self.N_trainrecords = self.df_train.shape[0]
		self.N_batches = int(self.N_trainrecords / BATCHSIZE)
		self.sparseness = np.mean((alldata.drop(['role','sound_ID'],axis=1).to_numpy()>1).astype(int),axis=None)
		self.SL = self.df_train.shape[1]-1 # sequence length
		self.VL = self.cell_IDs.shape[0]+1 # vocabulary length (including 0)
		print(self.df_train.shape[0],'train records')
		print(self.df_validation.shape[0],'validation records')
		print(self.df_test.shape[0],'test records')
		self.batchindexset = set(range(self.N_trainrecords))
		if self.cell_IDs.min()!=1 or self.cell_IDs.max()!=self.cell_IDs.shape[0]:
			print('error: cell IDs are not contiguous!')
			exit(-1)
		self.save_df_summary()

	def save_df_summary(self):
		f = open(os.path.join(OUTDIR, 'df_dummary.txt'),"w")
		f.write('N of stimuli: '+str(self.n_classes)+'\n')
		f.write('N of cells: '+str(self.cell_IDs.shape[0])+'\n')
		f.write('N of records: '+str(self.N_allrecords)+'\n')
		for sound_ID in self.sound_IDs:
			f.write('\t sound '+str(sound_ID)+': '+str(self.class_counts[sound_ID])+'\n')
		f.write('L: '+str(self.SL)+'\n')
		f.write('Sparseness: '+str(self.sparseness)+'\n')
		f.close()

	def get_batch(self):
		batch = []
		labels = []
		sampleindices = random.sample(self.batchindexset, BATCHSIZE)
		self.batchindexset = self.batchindexset - set(sampleindices)
		if len(self.batchindexset) < BATCHSIZE:
			self.batchindexset = set(range(self.N_trainrecords))
		df_batch = self.df_train.iloc[sampleindices]
		for row_idx, row in df_batch.iterrows():
			labels.append(row['sound_ID']-1)
			batch.append(row.drop('sound_ID').to_numpy(dtype=int))
		batch = {'input':torch.from_numpy(np.stack(batch)).to(self.device), 'label':torch.tensor(labels,dtype=int).to(self.device)}
		return batch

	def get_validation_batch(self): # return all validation records
		batch = []
		labels = []
		for row_idx, row in self.df_validation.iterrows():
			labels.append(row['sound_ID']-1)
			batch.append(row.drop('sound_ID').to_numpy(dtype=int))
		batch = {'input':torch.from_numpy(np.stack(batch)).to(self.device), 'label':torch.tensor(labels,dtype=int).to(self.device)}
		return batch

	def get_test_batch(self): # return all test records
		batch = []
		labels = []
		for row_idx, row in self.df_test.iterrows():
			labels.append(row['sound_ID']-1)
			batch.append(row.drop('sound_ID').to_numpy(dtype=int))
		batch = {'input':torch.from_numpy(np.stack(batch)).to(self.device), 'label':torch.tensor(labels,dtype=int).to(self.device)}
		return batch

	def get_firing_distribution_in_train_set(self):
		result = np.zeros(self.SL)
		for row_idx, row in self.df_train.iterrows():
			firings = np.sign(row.drop('sound_ID').to_numpy(dtype=int))
			result += firings
		result /= result.sum()
		return result

class GRU_network_attention(nn.Module):
	def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, n_classes):
		super(GRU_network_attention, self).__init__()
	
		self.cell_embedding = nn.Embedding(num_embeddings, embedding_size)
		
		self.gru = nn.GRU(EMBEDDING_SIZE, GRU_HIDDEN_SIZE, num_layers=GRU_LAYERS, bidirectional=BIDIRECTIONAL_GRU, batch_first=True)

		if BIDIRECTIONAL_GRU:
			self.num_directions = 2
		else:
			self.num_directions = 1

		self.W1 = torch.nn.Parameter(torch.Tensor(D_a, self.num_directions*GRU_HIDDEN_SIZE),requires_grad=True)

		linear_input_size = D_a * self.num_directions * GRU_HIDDEN_SIZE
		self.fc1 = nn.Linear(linear_input_size, FC_HIDDEN_SIZE)
		self.fc2 = nn.Linear(FC_HIDDEN_SIZE, n_classes)
	
	def forward(self, x, apply_softmax=False, returnall = False):
		batchsize = x.shape[0]
		batch_W1 = self.W1.unsqueeze(0).repeat(batchsize,1,1)
		x = self.cell_embedding(x)
		gru_output, gru_h_n = self.gru(x)
		x = torch.bmm(batch_W1, gru_output.permute(0,2,1))
		x = torch.tanh(x)
		attention_matrices = F.softmax(x,dim=2)
		sequence_embeddings = torch.bmm(attention_matrices, gru_output)
		x = torch.sigmoid(self.fc1(sequence_embeddings.reshape(batchsize,-1)))
		output = self.fc2(x)
		if apply_softmax:
			output = F.softmax(output, dim=1)
		if returnall:
			return output, sequence_embeddings, attention_matrices
		return output

def set_seed_everywhere(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(y_pred, y_target):
	_, y_pred_indices = y_pred.max(dim=1)
	n_correct = torch.eq(y_pred_indices, y_target).sum().item()
	return n_correct / len(y_pred_indices) * 100

def compute_rankdist(y_pred, y_target):
	_, y_pred_indices = y_pred.max(dim=1)
	dist = (y_pred_indices-y_target).abs().sum().item()/y_target.shape[0]
	return dist

def save_model(networkmodel,filename):
	modelfiles = [f for f in os.listdir(OUTDIR) if f.endswith('pt')]
	for modelfile in modelfiles:
		os.remove(os.path.join(OUTDIR,modelfile))
	torch.save(networkmodel.state_dict(), os.path.join(OUTDIR,filename))
	print('saving '+filename)

def get_saved_model():
	resultfiles = [f for f in os.listdir(OUTDIR) if f.endswith('pt')]
	filename = resultfiles[0]
	networkmodel = GRU_network_attention(mydataset.VL, EMBEDDING_SIZE, GRU_HIDDEN_SIZE, mydataset.n_classes)#.double()
	networkmodel.load_state_dict(torch.load(os.path.join(OUTDIR,filename)))
	networkmodel = networkmodel.to(mydevice)
	networkmodel.eval()
	print('loading '+filename)
	return networkmodel

cuda = torch.cuda.is_available()
mydevice = torch.device("cuda" if cuda else "cpu")
print("Using CUDA: {}".format(cuda))

set_seed_everywhere(RANDOMSEED, cuda)

mydataset = MyDataset(mydevice)
mydataset.get_batch()

mynetwork = GRU_network_attention(mydataset.VL, EMBEDDING_SIZE, GRU_HIDDEN_SIZE, mydataset.n_classes)#.double()
mynetwork = mynetwork.to(mydevice)

loss_func = nn.CrossEntropyLoss(mydataset.class_weights).to(mydevice)
if REGULARIZATION == 'L2':
	optimizer = optim.Adam(mynetwork.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
elif regularization == 'L1':
	pass
else:
	optimizer = optim.Adam(mynetwork.parameters(), lr=LEARNING_RATE)

result = {'train_loss':[], 'train_accuracy':[], 'val_loss':[], 'val_accuracy':[]}

# TRAINING WITH EVALUATION
min_eval_loss = None
best_eval_acc = None
for idx_epoch in range(N_EPOCHS):
	# TRAINING
	running_loss = 0.0
	running_acc = 0.0
	running_dist = 0.0
	mynetwork.train()
	for idx_batch in range(mydataset.N_batches):
		batch =  mydataset.get_batch()
		# step 1. zero the gradients
		optimizer.zero_grad()
		# step 2. compute the output
		y_pred = mynetwork(batch['input'])
		# step 3. compute the loss
		loss = loss_func(y_pred, batch['label'])
		loss_t = loss.item()
		running_loss += (loss_t - running_loss) / (idx_batch + 1)
		# step 4. use loss to produce gradients
		loss.backward()
		# step 5. use optimizer to take gradient step
		optimizer.step()
		# compute accuracy
		acc_t = compute_accuracy(y_pred, batch['label'])
		running_acc += (acc_t - running_acc) / (idx_batch + 1)
		# compute distance
		dist_t = compute_rankdist(y_pred, batch['label'])
		running_dist += (dist_t - running_dist) / (idx_batch + 1)
	result['train_loss'].append(running_loss)
	result['train_accuracy'].append(running_acc)
	# -----------------------------------------
	# EVALUATION
	mynetwork.eval()
	valbatch =  mydataset.get_validation_batch()
	# compute the output
	y_pred =  mynetwork(valbatch['input'])
	# compute the loss
	loss = loss_func(y_pred, valbatch['label']).item()
	# compute accuracy
	acc = compute_accuracy(y_pred,valbatch['label'])
	# save if best loss
	if min_eval_loss is None or loss<min_eval_loss:
		min_eval_loss = loss
		best_eval_acc = acc
		save_model(mynetwork,'model_'+str(idx_epoch+1)+'.pt')
	# compute distance
	dist = compute_rankdist(y_pred, valbatch['label'])
	result['val_loss'].append(loss)
	result['val_accuracy'].append(acc)
	#
	print('epoch',idx_epoch,'/',N_EPOCHS, result['train_loss'][-1], result['val_loss'][-1], result['train_accuracy'][-1], result['val_accuracy'][-1])

# save training process
result = pd.DataFrame(result)
result.to_csv(os.path.join(OUTDIR, 'result_training.csv'), index=False)
# load saved model
mynetwork = get_saved_model()
# save predictions, sequence embeddings, and attentions for the test records
mynetwork.eval()
testbatch = mydataset.get_test_batch()
y_pred, sequence_embeddings, attention_matrices = mynetwork(testbatch['input'],returnall=True)
# save predictions
s_real = pd.Series(testbatch['label'].cpu().numpy().astype(int))
_, y_pred_indices = y_pred.max(dim=1)
s_predicted = pd.Series(y_pred_indices.cpu().numpy())
df_pred = pd.DataFrame({'real':s_real, 'predicted':s_predicted})
df_pred.to_csv(os.path.join(OUTDIR, 'result_pred.csv'), index=False)
test_acc = (s_predicted==s_real).to_numpy().astype(int).mean()
print('Prediction accuracy on test set:',test_acc)
# save firing distribution
fd = mydataset.get_firing_distribution_in_train_set()
df_firing_distr = pd.DataFrame(fd,columns=['p'])
df_firing_distr.to_csv(os.path.join(OUTDIR, 'result_firingdistr.csv'), index=False)

f = open( os.path.join(OUTDIR, 'performance_summary.csv'),'w')
f.write('dataset, best eval loss, best eval acc, test acc\n')
f.write(DATASET+', '+str(min_eval_loss)+', '+str(best_eval_acc)+', '+str(test_acc)+'\n') 
f.close() 
