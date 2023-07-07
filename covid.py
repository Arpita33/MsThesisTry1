import pandas as pd
import os
import torch
import numpy as np
import utils #change it later
def get_data_min_max(records):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max
	
def variable_time_collate_fn(batch, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	combined_tt, all_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	#print(combined_tt, inverse_indices)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	
	combined_labels = None
	N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)
	
	#combined_recordIDs = torch.zeros(len(batch), N_labels)
	#combined_recordIDs = combined_recordIDs.to(device = device)
	combined_recordIDs =[]
	
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		#if record_id is not None:
		#	record_id = record_id.to(device)
		if labels is not None:
			#print(f"labels type: {type(labels)}")
			#print(labels)
			labels = labels.to(device)

		indices = all_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		if labels is not None:
			combined_labels[b] = labels
		if record_id is not None:
			combined_recordIDs.append(int(record_id))

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max)

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)
		
	combined_recordIDs = torch.tensor(combined_recordIDs, dtype=torch.int64)
		
	data_dict = {
		"record_ids":combined_recordIDs,
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}
	
	data_dict = utils.split_data_interp(data_dict)
	#data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict

class Covid(object):

    def __init__(self, type_,k_val,n_samples = None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        #self.data=pd.DataFrame(data_path)
        
        self.params = ['Gender', 'Race', 'Ethnicity','Age','HCT', 'WBC_NUM', 'BUN', 'CREAT', 'FLU_TEST', 'PLT', 'HGB',
       'BIL_TOT', 'ALT', 'ALKP', 'PROT_TOT', 'ALB_SER', 'AST', 'C19_TEST',
       'FIBR', 'LDH', 'PROCALC', 'FERR', 'LYMPH_NUM', 'LYMPH_PERC',
       'BIL_DIR', 'Troponin T', 'LAC_SER', 'INR', 'PT', 'CK', 'CRP',
       'D-Dimer FEU', 'ESR', 'CRP_S', 'D-Dimer DDU', 'aPTT', 'Troponin I',
       'D-Dimer', 'CORONA_TEST', 'IL6','AIDS/HIV', 'Any malignancy', 'Cerebrovascular disease',
       'Chronic pulmonary disease', 'Congestive heart failure', 'Dementia',
       'Diabetes with complications', 'Diabetes without complications',
       'Hemiplegia or paraplegia', 'Metastatic solid tumor',
       'Mild liver disease', 'Moderate or severe liver disease',
       'Myocardial infarction', 'Peptic ulcer disease',
       'Peripheral vascular disease', 'Renal disease', 'Rheumatic disease','CharlsonIndex']
        
        
        self.reduce = "average"
        self.Type = type_
        self.quantization = 0.1
        self.params_dict = {k: i for i, k in enumerate(self.params)}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #for key,value in self.params_dict.items():
        #    print(key,value)
        #self.processed_folder = "../../processed_data"
        #self.processed_folder = "./processed_data"
        #self.processed_folder = "./processed_new"
        #self.processed_folder = "../../processed_new"
        self.processed_folder  = "/N/project/C19Supp_2020/NingLab/Arpita/MGRU/Data/k_"+str(k_val)	
        if self.Type == "Train":
            #self.process("../../Cov_Data_Verified_Training",self.processed_folder,Train=True)
            #self.process("../../Train",self.processed_folder,extension = "train")
            data_file = self.training_file
        elif self.Type == "Valid":
            #self.process("../../Valid",self.processed_folder,extension = "validion")
            data_file = self.validation_file
        elif self.Type == "Test":
            #self.process("../../Cov_Data_Verified_Testing",self.processed_folder,Train=False)
            #self.process("../../Test",self.processed_folder,extension = "test")
            data_file = self.test_file
        print(device)
        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
            #self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, data_file))
            #self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            #self.labels = self.labels[:n_samples]

    #def get_dataset(self):
    #    return self.data


    # def process(self, raw_data_folder, processed_data_folder, extension):
    #     patients = []
    #     total = 0
    #     dirname = raw_data_folder
    #     for txtfile in os.listdir(dirname):
    #         record_id = txtfile.split('.')[0]
    #         label = None
    #         #print(f"{txtfile}")
    #         with open(os.path.join(dirname, txtfile)) as f:
    #             lines = f.readlines()
    #             if len(lines)==0:
    #                 continue
    #             label_line = lines[0].strip()
    #             #label = int(lines[0].split("=")[1])
    #             label = label_line.split("=")[1]
    #             label = np.array(label).astype(float)
    #             label = torch.tensor(label)
    #             #print(f"{txtfile} -> {label}")
    #             prev_time = 0
    #             tt = [0.]
    #             vals = [torch.zeros(len(self.params)).to(self.device)]
    #             mask = [torch.zeros(len(self.params)).to(self.device)]
    #             nobs = [torch.zeros(len(self.params))]
    #             for l in lines[2:]:
    #                 total += 1
    #                 time, param, val = l.split(',')
    #                 param = param.strip()
    #                 time = float(time)
    #                 time = round(time / self.quantization) * self.quantization

    #                 if time != prev_time:
    #                     tt.append(time)
    #                     vals.append(torch.zeros(len(self.params)).to(self.device))
    #                     mask.append(torch.zeros(len(self.params)).to(self.device))
    #                     nobs.append(torch.zeros(len(self.params)).to(self.device))
    #                     prev_time = time
                    
    #                 if param in self.params_dict:
    #                     n_observations = nobs[-1][self.params_dict[param]]
    #                     if self.reduce == 'average' and n_observations > 0:
    #                         prev_val = vals[-1][self.params_dict[param]]
    #                         new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
    #                         vals[-1][self.params_dict[param]] = new_val							                                                
    #                     else:
    #                         vals[-1][self.params_dict[param]] = float(val)
    #                     mask[-1][self.params_dict[param]] = 1
    #                     nobs[-1][self.params_dict[param]] += 1
    #                 else:
    #                     #print(f"here:{param}")
    #                     assert param == 'RecordID', 'Read unexpected param {}'.format(param)
            
    #         tt = torch.tensor(tt).to(self.device)
    #         vals = torch.stack(vals)
    #         mask = torch.stack(mask)

    #         patients.append((record_id, tt, vals, mask, label))
    #     #if Train:
    #     #    extension = "train"
    #     #else:
    #     #    extension = "test"
    #     torch.save(
	# 			patients,
	# 			os.path.join(processed_data_folder, 
	# 				'Covid_Processed_'+extension+'.pt')
	# 		)
    def process(self, curr_folder,extension,k_value):
        #curr_folder = train_folder
        files = os.listdir(curr_folder)
        patients = []
        for file in files:
            record_id = file.split('.')[0]
            f = open(curr_folder+file,"r")
            lines = f.readlines()
            label_line = lines[0].strip()
            label = int(lines[0].split("=")[1])
            label = label_line.split("=")[1]
            label = np.array(label).astype(float)
            label = torch.tensor(label)
            gender_line = lines[3].strip()
            gender = gender_line.split(",")[2]
            gender = gender.strip()
            race_line = lines[4].strip()
            race = race_line.split(",")[2]
            race = race.strip()
            eth_line = lines[5].strip()
            eth = eth_line.split(",")[2]
            age_line = lines[6].strip()
            age = age_line.split(",")[2]
            age = age.strip()
            prev_time = 0
            tt = [0.]
            vals = [torch.zeros(len(self.params))]
            mask = [torch.zeros(len(self.params))]
            for l in lines[7:]:
                time, param, val = l.split(',')
                param = param.strip()
                time = float(time)
                time = round(time / self.quantization) * self.quantization
                if time != prev_time:
                    tt.append(time)
                    vals.append(torch.zeros(len(self.params)))
                    mask.append(torch.zeros(len(self.params)))
                    prev_time = time

                vals[-1][self.params_dict['Gender']] = float(gender)
                mask[-1][self.params_dict['Gender']] = 1
                vals[-1][self.params_dict['Race']] = float(race)
                mask[-1][self.params_dict['Race']] = 1
                vals[-1][self.params_dict['Ethnicity']] = float(eth)
                mask[-1][self.params_dict['Ethnicity']] = 1
                vals[-1][self.params_dict['Age']] = float(age)
                mask[-1][self.params_dict['Age']] = 1
                if param in self.params_dict:
                    vals[-1][self.params_dict[param]] = float(val)
                    mask[-1][self.params_dict[param]] = 1
                else:
                    #print(f"here:{param}")
                    assert param == 'RecordID', 'Read unexpected param {}'.format(param)

            tt = torch.tensor(tt)
            vals = torch.stack(vals)
            mask = torch.stack(mask)
            patients.append((record_id, tt, vals, mask, label))

        processed_data_folder = "./k_folds/processed/k_"+str(k_value) 
        if not os.path.exists(processed_data_folder):
            os.mkdir(processed_data_folder)    
        torch.save(patients,os.path.join(processed_data_folder, 'Covid_Processed_'+extension+'.pt'))
    
    
    @property
    def training_file(self):
        return 'Covid_Processed_train.pt'

    @property
    def test_file(self):
        return 'Covid_Processed_test.pt'
    
    @property
    def validation_file(self):
        return 'Covid_Processed_valid.pt'    
        
    def __getitem__(self, index):
	    return self.data[index]
	    
    def __len__(self):
        return len(self.data)
		
                        

            
        

