from covid import *
from utils import *
from sklearn import model_selection
from torch.utils.data import DataLoader
from torch.nn.modules.rnn import GRUCell
import torch.nn as nn
import torch.nn.utils as nn_utils
import argparse
import time
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
import random

random.seed(42)


def create_Train_Val_Test_Dataloaders(batch_size,k):	
	train_dataset_obj = Covid(type_="Train",k_val = k, n_samples = 4000)
	valid_dataset_obj = Covid(type_="Valid",k_val = k, n_samples = 500)
	test_dataset_obj = Covid(type_="Test",k_val = k, n_samples = 500)
	total_dataset = train_dataset_obj[:len(train_dataset_obj)]+ valid_dataset_obj[:len(valid_dataset_obj)] +test_dataset_obj[:len(test_dataset_obj)]
	train_data,valid_data,test_data= train_dataset_obj, valid_dataset_obj, test_dataset_obj
	record_id, tt, vals, mask, labels = train_data[0]

	n_samples = len(total_dataset)
	input_dim = vals.size(-1)
	
	#batch_size = min(min(len(train_dataset_obj), batch_size), 5000)
	data_min, data_max = get_data_min_max(total_dataset)

	train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
	collate_fn= lambda batch: variable_time_collate_fn(batch, data_type = "train",data_min = data_min, data_max = data_max))

	valid_dataloader = DataLoader(valid_data, batch_size= batch_size, shuffle=False, 
	collate_fn= lambda batch: variable_time_collate_fn(batch, data_type = "train",data_min = data_min, data_max = data_max))


	test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
	collate_fn= lambda batch: variable_time_collate_fn(batch, data_type = "test",data_min = data_min, data_max = data_max))

	
	
	return train_dataloader, valid_dataloader, test_dataloader


class GRU_V1(nn.Module):
    def __init__(self, hidden,num_layers, dropout):
        super(GRU_V1,self).__init__()
        self.input_sz = 39*2+1
        self.hidden_sz = hidden#20
        self.layers = num_layers#5
        #self.batch_size = 50
        self.GRU = nn.GRU(input_size=self.input_sz, hidden_size=self.hidden_sz, num_layers=self.layers, batch_first=True, dropout = dropout)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_sz,1),nn.Sigmoid(),)
    
    def forward(self, data,batch_size,labels):
        all_hiddens = []
        hidden = torch.randn(self.layers,batch_size,self.hidden_sz) #(D*num_layers,N,H_out)
        output, hn = self.GRU(data,hidden)
        probs = self.classifier(output[:,-1,:])
        return probs

class GRU_V2(nn.Module):
    def __init__(self,batch_size, hidden,num_layers, dropout):
        super(GRU_V2,self).__init__()
        self.input_sz = 39*2+1
        self.hidden_sz = hidden
        self.layers = num_layers
        self.batch_size = batch_size
        self.GRU = nn.GRU(input_size=self.input_sz, hidden_size=self.hidden_sz, num_layers=self.layers, batch_first=True, dropout=dropout)

        self.classifier = nn.Linear(self.hidden_sz,2) #2 classes
        nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, data,labels):
        all_hiddens = []
        hidden = torch.randn(self.layers,self.batch_size,self.hidden_sz) #(D*num_layers,N,H_out)
        output, hn = self.GRU(data,hidden)
        probs = self.classifier(output[:,-1,:])
        return probs
        
class GRU_V3(nn.Module):
    def __init__(self, batch_size, hidden, num_layers, dropout):
        super(GRU_V3,self).__init__()
        #self.input_sz = 39*2+1
        self.input_sz = 58*2+1
        self.hidden_sz = hidden
        self.layers = num_layers
        self.batch_size = batch_size
        
        self.GRU = nn.GRU(input_size=self.input_sz, hidden_size=self.hidden_sz, num_layers=self.layers, batch_first=True,dropout=dropout)
        
        self.fc1 = nn.Linear(self.hidden_sz,self.hidden_sz//2)
        self.fc2 = nn.Linear(self.hidden_sz//2,self.hidden_sz//4)
        self.fc3 = nn.Linear(self.hidden_sz//4,2)
        self.classifier = nn.Sequential(self.fc1,nn.ReLU(),self.fc2,nn.ReLU(),self.fc3)
        #self.classifier = nn.Linear(self.hidden_sz,2) #2 classes
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        #nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, data,labels):
        all_hiddens = []
        hidden = torch.randn(self.layers,self.batch_size,self.hidden_sz) #(D*num_layers,N,H_out)
        output, hn = self.GRU(data,hidden)
        probs = self.classifier(output[:,-1,:])
        return probs

def get_data_and_labels(batch_dict, batch_size):
    truth_time_steps = batch_dict["observed_tp"]
    data=batch_dict["observed_data"]
    inputs = data
    mask = batch_dict["observed_mask"]
    delta_ts = truth_time_steps[1:]- truth_time_steps[:-1]
    zero_delta_t = torch.Tensor([0.])
    delta_ts = torch.cat((delta_ts, zero_delta_t))
    labels = batch_dict["labels"]
    n_steps = delta_ts.shape[0]
    if len(delta_ts.size()) == 1:
        delta_ts = delta_ts.unsqueeze(-1).repeat((batch_size,1,1))  
    
    inp1 = torch.cat((data, mask), -1)
    inp2 = torch.cat((inp1, delta_ts), -1) #N,L,H_in
    return inp2,labels


    
def train_one_epoch(model,optimizer,loss_fn, batch_size, train_dataloader):
    model.train(True)
    running_loss = 0.
    running_acc = 0
    last_loss = 0.
    total = 0
    correct = 0
    losses = []    
    
    min_clip_value = -3.0
    clip_value = 5.0
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    
    all_preds=[]
    all_labels = [] 
    #for i in range(num_batches):
    
    for i, batch_dict in enumerate(train_dataloader):
        #batch_dict = get_next_batch(data_objects["train_dataloader"])
        #batch_dict = train_objects[i]
        truth_time_steps = batch_dict["observed_tp"]
        data=batch_dict["observed_data"]
        inputs = data
        mask = batch_dict["observed_mask"]
        delta_ts = truth_time_steps[1:]- truth_time_steps[:-1]
        zero_delta_t = torch.Tensor([0.])
        delta_ts = torch.cat((delta_ts, zero_delta_t))
        labels = batch_dict["labels"].squeeze()
        labels = labels.long()
        #print(sum(labels))
        n_steps = delta_ts.shape[0]
        if len(delta_ts.size()) == 1:
            #print("yes")
            delta_ts = delta_ts.unsqueeze(-1).repeat((batch_size,1,1))        
        inp1 = torch.cat((data, mask), -1)
        #print(inp1.shape)
        #sys.stdout.flush()
        inp2 = torch.cat((inp1, delta_ts), -1) #N,L,H_in

        # Make predictions for this batch
        #outputs = model(inputs)
        #train_batch_size = 100
        outputs = model(inp2,labels)
        probs = torch.softmax(outputs,dim = 1)

        _, predicted = torch.max(outputs.data, 1)
        #probabilities,indices = torch.max(probs.data, 1)
        probabilities = probs[:,1]

        if i==0:
            all_preds = predicted
            all_probs = probabilities
            all_labels = labels
        else:
            all_preds = torch.cat((all_preds,predicted),-1)
            all_probs = torch.cat((all_probs,probabilities),0)
            all_labels = torch.cat((all_labels,labels),-1)
            
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        #print(labels)
        loss = loss_fn(outputs, labels)        
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        losses.append(loss.item())
    
    last_loss = running_loss/(i+1)    
    #last_loss = running_loss/(num_batches)
    avg_acc = correct/total

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs.detach().numpy())

    return last_loss, accuracy, precision, recall, f1, auc    
    

def train(hidden_sz, layers, dropout, batch_size, learning_rate, num_epochs, exp_no, k_val):
	train_dataloader,valid_dataloader, test_dataloader = create_Train_Val_Test_Dataloaders(batch_size,k_val)
	model = GRU_V3(batch_size, hidden_sz, layers, dropout)
	# Optimizers specified in the torch.optim package
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #was 0.01
	loss_fn = torch.nn.CrossEntropyLoss()#torch.nn.BCELoss()
	epoch_number = 0
	EPOCHS = num_epochs
	train_losses = []
	best_valid_auc = 0
	#exp_no = 1
	for epoch in range(EPOCHS):
		start = time.time()
		print('EPOCH {}:'.format(epoch_number + 1))
		#avg_loss, avg_acc = train_one_epoch(model)
		avg_loss, t_accuracy, t_precision, t_recall, t_f1, t_auc = train_one_epoch(model,optimizer,loss_fn, batch_size, train_dataloader)
		train_losses.append(avg_loss)
		#print('Epoch {} loss: {}'.format(epoch + 1, avg_loss))
		
		model.train(False)

		running_vloss = 0.0
		for i, vdata in enumerate(valid_dataloader):
			vinputs, vlabels = get_data_and_labels(vdata,batch_size)
			vlabels = vlabels.squeeze()
			vlabels = vlabels.long()  
			voutputs = model(vinputs,vlabels)
			vloss = loss_fn(voutputs, vlabels)
			running_vloss += vloss
			_, vpreds = torch.max(voutputs.data, 1)

			vprobs = torch.softmax(voutputs,dim = 1)
			vprobabilities = vprobs[:,1]
			#vprobabilities,indices = torch.max(vprobs.data, 1)

			if i==0:
				all_preds_v = vpreds
				all_probs_v = vprobabilities
				all_labels_v = vlabels
			else:
				all_preds_v = torch.cat((all_preds_v,vpreds),-1)
				all_probs_v = torch.cat((all_probs_v,vprobabilities),0)
				all_labels_v = torch.cat((all_labels_v,vlabels),-1)
			
		v_accuracy = accuracy_score(all_labels_v, all_preds_v)
		v_precision = precision_score(all_labels_v, all_preds_v)
		v_recall = recall_score(all_labels_v, all_preds_v)
		v_f1 = f1_score(all_labels_v, all_preds_v)
		v_auc = roc_auc_score(all_labels_v, all_probs_v.detach().numpy())    
  
		avg_vloss = running_vloss / (i + 1)
		time_for_one_epoch = time.time() - start
		
		
		print('Time: {}'.format(time_for_one_epoch))
		print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
		#print('Accuracy train: {}, valid: {}'.format(avg_acc,total_v_acc ))
		print('Accuracy train: {}, valid: {}'.format(t_accuracy,v_accuracy ))
		print('Precision train: {}, valid: {}'.format(t_precision,v_precision ))
		print('Recall train: {}, valid: {}'.format(t_recall,v_recall))
		print('F1 train: {}, valid: {}'.format(t_f1,v_f1 ))
		print('AUROC train: {}, valid: {}'.format(t_auc,v_auc))
		sys.stdout.flush()
		
		if v_auc>best_valid_auc:
			best_valid_auc = v_auc
			##perform evaluation on test set here
			print()
			print("Performing evaluation on the Test Set")
			test_loss, test_acc, test_pr, test_re, test_f1, test_auc = evaluate_model(model,loss_fn,test_dataloader,batch_size)
			print('Test Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}, AUC: {}'.format(test_loss, test_acc, test_pr, test_re, test_f1, test_auc))
			print()
			sys.stdout.flush()
			model_folder = "/N/project/C19Supp_2020/NingLab/Arpita/Another Run/Models/k_"+str(k_val)+"/"
			model_path = model_folder+'model_{}_{}'.format(exp_no, epoch_number)
			torch.save(model.state_dict(), model_path)
		
		epoch_number += 1

def evaluate_model(model,loss_fn,data_loader,batch_size):
    running_loss=0

    for i, data in enumerate(data_loader):
        inputs, labels = get_data_and_labels(data,batch_size)
        labels = labels.squeeze()
        labels = labels.long()  
        outputs = model(inputs,labels)
        loss = loss_fn(outputs, labels)
        running_loss += loss
        _, preds = torch.max(outputs.data, 1)
        
        probs = torch.softmax(outputs,dim = 1)
        #probabilities,indices = torch.max(probs.data, 1)
        probabilities = probs[:,1]

        if i==0:
            all_preds = preds
            all_probs = probabilities
            all_labels = labels
        else:
            all_preds = torch.cat((all_preds,preds),-1)
            all_probs = torch.cat((all_probs,probabilities),0)
            all_labels = torch.cat((all_labels,labels),-1)
        
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs.detach().numpy())       
    avg_loss = running_loss / (i + 1)
    
    return avg_loss, accuracy, precision, recall, f1, auc
		

parser = argparse.ArgumentParser('')
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--hidden-dims', type=int, default=20, help="Size of GRU Hidden Layer.")
parser.add_argument('--num-layers', type=int, default=2, help="Number of layers in multi-layered GRU.")
parser.add_argument('--dropout', type=float, default=0, help="Dropout probability after GRU Layers.")
parser.add_argument('--lr',  type=float, default=1e-2, help="learning rate.")
parser.add_argument('--epochs',  type=int, default=100, help="Number of training epochs.")
parser.add_argument('--exp-no',  type=int,default=0, help="Experiment Number")
parser.add_argument('--k',type=int,default=0, help="K-fold number")


args = parser.parse_args()
		
def main():
    #train(args.hidden_dims, args.num_layers, args.dropout, args.batch_size, args.lr, args.epochs)
    train(args.hidden_dims, args.num_layers, args.dropout, args.batch_size, args.lr, args.epochs, args.exp_no, args.k)
    

if __name__ == "__main__":
    main()

