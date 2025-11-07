import os
import sys
import argparse
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = './data' #os.environ['HOME'] + '/data'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#region ms_data

def min_max_scale(x, min, max):
    new_x = 1.0 * (x - min) / (max - min)
    return new_x

def min_max_scale_rev(x, min, max):
    old_x = x * (max - min) + min
    return old_x

class data_phospho():

    @classmethod
    def load_training_transformer(cls):
        
        x_train = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_x_train.npy')
        y_train = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_y_train.npy')

        x_val = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_x_val.npy')
        y_val = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_y_val.npy')

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_testing_transformer(cls):
        x_test = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_x_test.npy')
        y_test = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_y_test.npy')

        return (x_test, y_test)

class data_autort():

    @classmethod
    def load_training_transformer(cls, max_sequence_length = 48):
        
        x_train = np.load(DATA_DIR + '/PXD006109_rt_x_train.npy')
        y_train = np.load(DATA_DIR + '/PXD006109_rt_y_train.npy')

        x_val = np.load(DATA_DIR + '/PXD006109_rt_x_val.npy')
        y_val = np.load(DATA_DIR + '/PXD006109_rt_y_val.npy')

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_testing_transformer(cls, max_sequence_length = 48):
        x_test = np.load(DATA_DIR + '/PXD006109_rt_x_test.npy')
        y_test = np.load(DATA_DIR + '/PXD006109_rt_y_test.npy')

        return (x_test, y_test)


class data_deepdia():

    @classmethod
    def load_training_transformer(cls, max_sequence_length = 50):

        x_train = np.load(DATA_DIR + '/PXD005573_rt_x_train.npy')
        y_train = np.load(DATA_DIR + '/PXD005573_rt_y_train.npy')

        x_val = np.load(DATA_DIR + '/PXD005573_rt_x_val.npy')
        y_val = np.load(DATA_DIR + '/PXD005573_rt_y_val.npy')

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_testing_transformer(cls, max_sequence_length = 50):

        x_test = np.load(DATA_DIR + '/PXD005573_rt_x_test.npy')
        y_test = np.load(DATA_DIR + '/PXD005573_rt_y_test.npy')

        return (x_test, y_test)

class data_prosit():
    @classmethod
    def load_training(cls, max_sequence_length = 30):

        x_train = np.load(DATA_DIR + '/prosit_rt_updated/X_train.npy')
        y_train = np.load(DATA_DIR + '/prosit_rt_updated/Y_train.npy')

        x_val = np.load(DATA_DIR + '/prosit_rt_updated/X_validation.npy')
        y_val = np.load(DATA_DIR + '/prosit_rt_updated/Y_validation.npy')

        return (x_train, y_train), (x_val, y_val)


    @classmethod
    def load_testing(cls, max_sequence_length = 30):

        x_test = np.load(DATA_DIR + '/prosit_rt_updated/X_holdout.npy')
        y_test = np.load(DATA_DIR + '/prosit_rt_updated/Y_holdout.npy')
        
        return (x_test, y_test)

class data_generics():
    
    @classmethod
    def sequence_to_integer(cls, sequences, max_sequence_length):

        Prosit_ALPHABET = {
            'A': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 7,
            'I': 8,
            'K': 9,
            'L': 10,
            'M': 11,
            'N': 12,
            'P': 13,
            'Q': 14,
            'R': 15,
            'S': 16,
            'T': 17,
            'V': 18,
            'W': 19,
            'Y': 20,
            'o': 21
        }

        array = np.zeros([len(sequences), max_sequence_length], dtype=int)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(re.sub(r'M\(ox\)', 'o', sequence)):
                array[i, j] = Prosit_ALPHABET[s]
        return array

    @classmethod
    def load_deepdia(cls, filename, seq_header = 'sequence', rt_header = 'rt'):

        min_sequence_length = 7
        max_sequence_length = 50

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d

        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header][1:-1] # remove _xxx_

            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row[rt_header]
                else :
                    selected_peptides[s] = 0

        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = [rt_header])
        x = cls.sequence_to_integer(df.index.values, max_sequence_length)

        return (x, df[rt_header].to_numpy())


    @classmethod
    def load_prosit(cls, filename, seq_header = 'sequence', rt_header = 'rt'):

        min_sequence_length = 7
        max_sequence_length = 30

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d

        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header][1:-1] # remove _xxx_
            s = re.sub(r'M\(ox\)', 'o', s)
            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row['rt']
                else :
                    selected_peptides[s] = 0


        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])
        x = cls.sequence_to_integer(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())

    @classmethod
    def load_autort(cls, filename, seq_header = 'sequence', rt_header = 'rt'):

        min_sequence_length = 7
        max_sequence_length = 48

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d


        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header][1:-1] # remove _xxx_
            s = re.sub(r'M\(ox\)', 'o', s)
            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row[rt_header]
                else :
                    selected_peptides[s] = 0


        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])
        x = cls.sequence_to_integer(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())

    @classmethod
    def integer_to_sequence(cls, X):
        int2seq = '_ACDEFGHIKLMNPQRSTVWYo'
        int2seqf = lambda x : ''.join([int2seq[c] for c in x if c > 0])
        return ([int2seqf(x) for x in X])

    @classmethod
    def sequence_to_integer_phospho(cls, sequences, max_sequence_length = 60):

        deepphospho_ALPHABET = {
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
            "1": 21,
            "2": 22,
            "3": 23,
            "4": 24,
            "*": 25
        }

        array = np.zeros([len(sequences), max_sequence_length], dtype=int)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(re.sub('@', '', sequence)):
                array[i, j] = deepphospho_ALPHABET[s]
        return array

    @classmethod
    def load_phospho(cls, filename, seq_header = 'IntPep', rt_header = 'iRT'):

        min_sequence_length = 6
        max_sequence_length = 60

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d


        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header]
            s = re.sub('@', '', s)
            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row[rt_header]
                else :
                    selected_peptides[s] = 0
            else:
                print(s)


        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])

        x = cls.sequence_to_integer_phospho(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())

    @classmethod
    def integer_to_sequence_phospho(cls, X):
        int2seq = '_ACDEFGHIKLMNPQRSTVWY1234*'
        int2seqf = lambda x : ''.join([int2seq[c] for c in x if c > 0])
        return ([int2seqf(x) for x in X])
    
def load_test_data(data, input_file, seq_header, rt_header, CLS, seq_length):
    if input_file is not None:
        if data == 'prosit':
            x_test, y_test = data_generics.load_prosit(input_file, seq_header, rt_header)                
        elif data == 'deepdia':
            x_test, y_test = data_generics.load_deepdia(input_file, seq_header, rt_header)               
        elif data == 'autort':
            x_test, y_test = data_generics.load_autort(input_file, seq_header, rt_header)
        elif data == 'phospho':
            x_test, y_test = data_generics.load_phospho(input_file, seq_header, rt_header)
        else:
            print('Unknown model')
            exit(0)
    elif data == 'prosit':
        x_test, y_test = data_prosit.load_testing()                
    elif data == 'deepdia':
        x_test, y_test = data_deepdia.load_testing_transformer()               
    elif data == 'autort':
        x_test, y_test = data_autort.load_testing_transformer()
    elif data == 'phospho':
        x_test, y_test = data_generics.load_phospho(input='', seq_header = 'sequence')
    else:
        print('Unknown model')
        exit(0)

    if (data == 'phospho'):
        all_peps = data_generics.integer_to_sequence_phospho(x_test)
    else:    
        all_peps = data_generics.integer_to_sequence(x_test)

    print(seq_length, x_test.shape)
    assert seq_length >= x_test.shape[1]
    x_test = np.concatenate((np.full((x_test.shape[0], 1), CLS), x_test), axis = 1)    
    if seq_length > x_test.shape[1]:        
        x_test = np.concatenate((x_test, np.full((x_test.shape[0],  (seq_length - x_test.shape[1])), CLS)), axis = 1)
    assert seq_length == x_test.shape[1]
    return torch.tensor(x_test), torch.tensor(y_test).unsqueeze(1), all_peps

class DatasetRT():
    def __init__(self, dataset, batch_size, epochs, device_type = 'cpu', device = 'cpu'):        
        if dataset == 'autort':
            (x_train, y_train), (x_val, y_val) = data_autort.load_training_transformer()

            min_val = 0.0
            max_val = 101.33

        elif dataset == 'prosit':
            (x_train, y_train), (x_val, y_val) = data_prosit.load_training()
            
            min_val = min(y_train.min(), y_val.min()) - 0.01
            max_val = max(y_train.max(), y_val.max()) + 0.01

            y_train = min_max_scale(y_train, min = min_val, max = max_val)
            y_val = min_max_scale(y_val, min = min_val, max = max_val)

        elif dataset == 'deepdia':
            (x_train, y_train), (x_val, y_val) = data_deepdia.load_training_transformer()
            
            min_val = min(y_train.min(), y_val.min()) - 0.01
            max_val = max(y_train.max(), y_val.max()) + 0.01            

            y_train = min_max_scale(y_train, min = min_val, max = max_val)
            y_val = min_max_scale(y_val, min = min_val, max = max_val)

        elif dataset == 'phospho':
            (x_train, y_train), (x_val, y_val) = data_phospho.load_training_transformer()

            min_val = 0.0
            max_val = 1.0

        else:
            print('Unknown data')
            exit(0)

        print("(min, max) = ", (min_val, max_val))  

        self.CLS = x_train.max() + 1
        x_train = np.concatenate((np.full((x_train.shape[0], 1), self.CLS), x_train), axis = 1)
        x_val = np.concatenate((np.full((x_val.shape[0], 1), self.CLS), x_val), axis = 1)

        train_tensor = TensorDataset(torch.tensor(x_train), torch.tensor(y_train).unsqueeze(1))        
        self.loader_train = DataLoader(train_tensor, batch_size=batch_size, shuffle=True,drop_last=True)
        print(f"Total number of train batches: {len(self.loader_train)}")
        
        val_tensor = TensorDataset(torch.tensor(x_val), torch.tensor(y_val).unsqueeze(1))
        self.loader_val = DataLoader(val_tensor, batch_size=batch_size, shuffle=True,drop_last=True)
        print(f"Total number of val batches: {len(self.loader_val)}")

        print('CLS =', self.CLS)
        # if training == True:
        #     print(len(x_train), 'Training sequences')            
        #     print(x_train.shape)
        #     print(y_train.shape)
        #     t = TensorDataset(x_train, y_train)            
        # else:
        #     print(len(x_val), 'Validation sequences')
        #     t = TensorDataset(x_val, y_val)
        
        self.train_iter = iter(self.loader_train)
        self.val_iter =  iter(self.loader_val)
        self.min_val = min_val
        self.max_val = max_val
        self.block_size = x_train.shape[1]
        self.vocab_size = self.CLS + 1
        self.device_type = device_type
        self.device = device
        self.epoch = epochs        

    def get_batch(self, split):
        if split == 'train':
            try:
                x,y = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.loader_train)   
                print(f"end of epoch {self.epoch}")
                self.epoch -= 1         
                x,y = next(self.train_iter)
        else:
            try:
                x,y = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.loader_val) 
                x,y = next(self.val_iter)

        if self.device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    

class DataLoaderRT:
    def __init__(self, dataset, batch_size, device_type = 'cpu', device = 'cpu'):        
        if dataset == 'autort':
            (x_train, y_train), (x_val, y_val) = data_autort.load_training_transformer()

            min_val = 0.0
            max_val = 101.33

        elif dataset == 'prosit':
            (x_train, y_train), (x_val, y_val) = data_prosit.load_training()
            
            min_val = min(y_train.min(), y_val.min()) - 0.01
            max_val = max(y_train.max(), y_val.max()) + 0.01

            y_train = min_max_scale(y_train, min = min_val, max = max_val)
            y_val = min_max_scale(y_val, min = min_val, max = max_val)

        elif dataset == 'deepdia':
            (x_train, y_train), (x_val, y_val) = data_deepdia.load_training_transformer()
            
            min_val = min(y_train.min(), y_val.min()) - 0.01
            max_val = max(y_train.max(), y_val.max()) + 0.01            

            y_train = min_max_scale(y_train, min = min_val, max = max_val)
            y_val = min_max_scale(y_val, min = min_val, max = max_val)

        elif dataset == 'phospho':
            (x_train, y_train), (x_val, y_val) = data_phospho.load_training_transformer()

            min_val = 0.0
            max_val = 1.0

        else:
            print('Unknown data')
            exit(0)

        print("(min, max) = ", (min_val, max_val))  

        self.batch_size = batch_size             
        self.next_batch_start = 0
        self.next_val_batch_start = 0
        self.device_type = device_type
        self.device = device
          
        self.min_val = min_val
        self.max_val = max_val
        
        self.CLS = x_train.max() + 1
        self.vocab_size = self.CLS
        self.y_train = y_train
        self.y_val = y_val        
        self.x_train = np.concatenate((np.full((x_train.shape[0], 1), self.CLS), x_train), axis = 1)
        self.x_val = np.concatenate((np.full((x_val.shape[0], 1), self.CLS), x_val), axis = 1)
        self.block_size = self.x_train.shape[1]
        print(len(self.x_train), 'Training sequences')
        print(len(self.x_val), 'Validation sequences')

        print('CLS =', self.CLS)
    
    def next_batch(self, split='train'):                
        if split == 'train':
            start = self.next_batch_start
            self.next_batch_start = min(len(self.x_train), start + self.batch_size)
            x = torch.stack([torch.from_numpy(self.x_train[i]) for i in range (start, self.next_batch_start)])
            y = torch.from_numpy(self.y_train[start : self.next_batch_start])
        else:
            start = self.next_val_batch_start
            self.next_val_batch_start = min(len(self.x_val), start + self.batch_size)
            x = torch.stack([torch.from_numpy(self.x_val[i]) for i in range (start, self.next_val_batch_start)])
            y = torch.from_numpy(self.y_val[start : self.next_val_batch_start])
        
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y