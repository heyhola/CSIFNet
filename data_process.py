import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import tifffile
import imgaug.augmenters as iaa
# import torchvision.transforms as transforms
from skimage.transform import resize
from omics.Reactome import *
from .utils import *

save_path = '/public1/home/xsq/gd/full_metabric/all_image_omics'
genes_path = '/public1/home/xsq/gd/omics/all_image'

class DatasetLIDC(data.Dataset):
    def __init__(self, data_sources, fold_id, fold_mean, fold_std, in_h, in_w,
                 fold_means_full, fold_std_full, fold_means_other, fold_std_other,
                 di_ct, di_rt, di_ht, di_er, cont_cols, cat_cols, n_classes_cat,
                 isTraining=True,isValidation = False):

        self.img_data_dir = data_sources.img_data_dir
        self.fold_splits = data_sources.fold_splits
        self.labels_file = data_sources.labels_file
        self.fold_id = fold_id
        self.fold_mean = fold_mean
        self.fold_std = fold_std
        self.in_height = in_h 
        self.in_width = in_w
        self.fold_means_full = fold_means_full
        self.fold_std_full = fold_std_full
        self.fold_means_other = fold_means_other
        self.fold_std_other = fold_std_other
        self.isTraining = isTraining

        self.di_ct = di_ct
        self.di_rt = di_rt 
        self.di_ht = di_ht
        self.di_er = di_er
        self.n_classes_cat = n_classes_cat

        # Data files
        if isTraining:
            self.img_id = np.load(self.fold_splits + '/' + str(self.fold_id) + '_train.npy')

        elif isValidation:
            self.img_id = np.load(self.fold_splits + '/' + str(self.fold_id) + '_validation.npy')

        else:
            self.img_id = np.load(self.fold_splits + '/' + str(self.fold_id) + '_test.npy')

        patient_subset_txt = self.fold_splits + '/all.txt'
        patient_subset_norm = patient_subset_txt
        self.img_paths = get_data_dirs_split(patient_subset_txt, self.img_data_dir)
        path = [os.path.basename (i) for i in self.img_paths]
        self.path = np.array(list(set(path)))
        all_sample = np.load(self.fold_splits + '/' + 'all_sample.npy')
        self.all_index = {i:[] for i in set(all_sample)}
        for i in range(len(self.path)):
            self.all_index[self.path[i][0:6]].append(i)
        
        self.img_paths_norm = get_data_dirs_split(patient_subset_norm, self.img_data_dir)
        self.img_paths_norm = [os.path.basename(x).split('_')[0] for x in self.img_paths_norm]
        self.img_paths_norm = [x[:2] + '-' + x[2:] for x in self.img_paths_norm]

        # All the labels for this data subset
        '''omics'''
        data_df = pd.read_csv(self.labels_file)
        data_df = data_df.drop(['Unnamed: 0'],axis = 1)
        data_df = data_df.rename(columns={'Patient_ID':'METABRIC.ID'})
        order = ['METABRIC.ID','label','OS_MONTHS','Age.At.Diagnosis','ER.Status','CT','RT','HT','ERBB2','PGR','EGFR','Ki67']
        data_df = data_df[order]
        data_df = data_df[~data_df['label'].isnull()]
        self.labels_df = data_df.iloc[:,:3]
        # get label
        self.label = data_df['label']     
        clinical_df = data_df[cont_cols]
        clinical_df_cat = data_df[cat_cols]
        
        self.clinical_df = clinical_df*1
        self.clinical_df_cat = clinical_df_cat*1
        
        # Normalise continuous vars: Age.At.Diagnosis, Ki67, EGFR, PR, HER2
        norm_df = self.clinical_df[self.labels_df['METABRIC.ID'].isin(self.img_paths_norm)]
        self.clinical_df["Age.At.Diagnosis"] = (clinical_df["Age.At.Diagnosis"]-norm_df["Age.At.Diagnosis"].mean())/norm_df["Age.At.Diagnosis"].std()
        self.clinical_df["Ki67"] = (clinical_df["Ki67"]-norm_df["Ki67"].mean())/norm_df["Ki67"].std()
        self.clinical_df["EGFR"] = (clinical_df["EGFR"]-norm_df["EGFR"].mean())/norm_df["EGFR"].std()
        self.clinical_df["PGR"] = (clinical_df["PGR"]-norm_df["PGR"].mean())/norm_df["PGR"].std()
        self.clinical_df["ERBB2"] = (clinical_df["ERBB2"]-norm_df["ERBB2"].mean())/norm_df["ERBB2"].std()
        
        # Map categorical to one-hot
        self.clinical_df_cat['CT'] = clinical_df_cat['CT']
        self.clinical_df_cat['RT'] = clinical_df_cat['RT']
        self.clinical_df_cat['HT'] = clinical_df_cat['HT']
        self.clinical_df_cat['ER.Status'] = clinical_df_cat['ER.Status']

        # get omics data
        cnv_del = pd.read_csv(join(genes_path + str('/cnv_del.csv')))
        cnv_del.index = cnv_del['Unnamed: 0'].values
        cnv_del = cnv_del.drop(['Unnamed: 0'],axis = 1)
        self.cnv_del = cnv_del

        cnv_amp = pd.read_csv(join(genes_path + str('/cnv_amp.csv')))
        cnv_amp.index=cnv_amp['Unnamed: 0'].values
        cnv_amp = cnv_amp.drop(['Unnamed: 0'],axis = 1)
        self.cnv_amp = cnv_amp

        mut = pd.read_csv(join(genes_path + str('/mut_data.csv')))
        mut.index = mut['Unnamed: 0'].values
        mut = mut.drop(['Unnamed: 0'],axis = 1)
        self.mut = mut
       
    def augment_data(self, batch_raw):
        batch_raw = np.expand_dims(batch_raw, 0)

        # Original, horizontal
        random_flip = np.random.randint(2, size=1)[0]
        # 0, 90, 180, 270
        random_rotate = np.random.randint(4, size=1)[0]

        # Flips
        if random_flip == 0:
            batch_flip = batch_raw*1
        else:
            batch_flip = iaa.Flipud(1.0)(images=batch_raw)
                
        # Rotations
        if random_rotate == 0:
            batch_rotate = batch_flip*1
        elif random_rotate == 1:
            batch_rotate = iaa.Rot90(1, keep_size=True)(images=batch_flip)
        elif random_rotate == 2:
            batch_rotate = iaa.Rot90(2, keep_size=True)(images=batch_flip)
        else:
            batch_rotate = iaa.Rot90(3, keep_size=True)(images=batch_flip)
        
        images_aug_array = np.array(batch_rotate)

        return images_aug_array


    def normalise_images(self, imgs, fold_mean, fold_std):        
        return (imgs - fold_mean)/fold_std


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.img_id)


    def __getitem__(self, index):
            img = np.load(save_path + '/'  + str(self.img_id[index]) + '/img.npy')
            clinical = np.load(save_path + '/' + str(self.img_id[index]) + '/clinical.npy')
            clinical[1],clinical[2],clinical[3],clinical[4] = clinical[2],clinical[3],clinical[4],clinical[1]
            clinical_cat = np.load(save_path + '/' + str(self.img_id[index]) + '/clinical_cat.npy')
            labels_time = np.load(save_path + '/' + str(self.img_id[index]) + '/labels_time.npy')
            labels_censored = np.load(save_path + '/' + str(self.img_id[index]) + '/labels_censored.npy')
            ID = str(np.load(save_path + '/' + str(self.img_id[index]) + '/ID.npy'))
            mut = np.load(save_path + '/'  + str(self.img_id[index]) + '/mut.npy')
            cnv_del = np.load(save_path + '/'  + str(self.img_id[index]) + '/cnv_del.npy')
            cnv_amp = np.load(save_path + '/'  + str(self.img_id[index]) + '/cnv_amp.npy')

            # Convert to tensor
            img_torch = torch.from_numpy(img).float()
            clinical_torch = torch.from_numpy(clinical).float()
            clinical_cat_torch = torch.from_numpy(clinical_cat).long()
            labels_torch = torch.from_numpy(labels_time).float()
            censored_torch = torch.from_numpy(labels_censored).long()
            mut = torch.from_numpy(mut).float()
            cnv_del = torch.from_numpy(cnv_del).float()
            cnv_amp = torch.from_numpy(cnv_amp).float()
            
            if self.isTraining:
                return img_torch, clinical_torch, clinical_cat_torch, labels_torch, censored_torch, mut, cnv_del, cnv_amp
            else:
                return img_torch, clinical_torch, clinical_cat_torch, labels_torch, censored_torch, ID, mut, cnv_del, cnv_amp

    def save_data(self, index):
        # 'Generates one sample of data'
        img_path = []
        for i in range(len(self.all_index[index])):
            path = join(self.img_data_dir,self.path[self.all_index[index][i]])
            img_path.append(path)
        img_path.sort(key = lambda i:os.path.basename(i).split('_')[2],reverse=True)
        img_shape = np.zeros((176,176,1))
        for i in range(len(img_path)):
            ID = os.path.basename(img_path[i])
            img = tifffile.imread(img_path[i])
            if len(img.shape) ==2:
                img = np.moveaxis(img, 0, -1)
                img = resize(img,(176,176),)
                assert(img.shape[0] == self.in_height)
                assert(img.shape[1] == self.in_width)
                img = self.normalise_images(img, self.fold_means_other, self.fold_std_other)
                img = np.expand_dims(img,2)
            elif img.shape[0]==39:
                img = np.moveaxis(img, 0, -1)
                img = resize(img,(176,176),)
                assert(img.shape[0] == self.in_height)
                assert(img.shape[1] == self.in_width)     
                img = self.normalise_images(img, self.fold_means_full, self.fold_std_full)
            elif img.shape[0]==50:
                img = np.moveaxis(img, 0, -1)
                img = resize(img,(176,176),)
                assert(img.shape[0] == self.in_height)
                assert(img.shape[1] == self.in_width)     
                img = self.normalise_images(img, self.fold_mean, self.fold_std)
            img_shape = np.concatenate((img_shape,img),axis=2)
        img = img_shape[:,:,1:] 

        if self.isTraining:
            img = np.squeeze(self.augment_data(img))
        
        img = np.moveaxis(img, -1, 0)

        # Get labels
        ID_reformat = ID.split('_')[0]
        ID_reformat = ID_reformat[:2] + '-' + ID_reformat[2:]
        labels = self.labels_df.loc[self.labels_df['METABRIC.ID'] == ID_reformat].values.tolist()

        labels_time = np.zeros(1)
        labels_censored = np.zeros(1)

        labels_time[0] = labels[0][-1]
        labels_censored[0] = int(labels[0][-2])

        clinical = self.clinical_df.loc[self.labels_df['METABRIC.ID'] == ID_reformat].values.tolist()
        
        # Scalar, 5-dim
        clinical = np.array(clinical[0])
        # One-hot, 4-dim        
        clinical_cat = self.clinical_df_cat.loc[self.labels_df['METABRIC.ID'] == ID_reformat].values.tolist()

        ct_1h = np.zeros(self.n_classes_cat[0])
        ct_1h[clinical_cat[0][0]] = 1
        rt_1h = np.zeros(self.n_classes_cat[1])
        rt_1h[clinical_cat[0][1]] = 1
        ht_1h = np.zeros(self.n_classes_cat[2])
        ht_1h[clinical_cat[0][2]] = 1
        er_1h = np.zeros(self.n_classes_cat[3])
        er_1h[clinical_cat[0][3]] = 1

        clinical = np.concatenate((clinical, ct_1h, rt_1h, ht_1h, er_1h))
        
        clinical_cat = np.array(clinical_cat[0])
       
        # Convert to tensor
        #####get omic datas######
        mut = self.mut.loc[self.mut.index == ID_reformat].values.tolist()
        cnv_del = self.cnv_del.loc[self.cnv_del.index == ID_reformat].values.tolist()
        cnv_amp = self.cnv_amp.loc[self.cnv_amp.index == ID_reformat].values.tolist()
       
        os.makedirs(save_path + '/' + str(index))
        np.save(save_path + '/' + str(index) + '/img',img)
        np.save(save_path + '/' + str(index) + '/clinical',clinical)
        np.save(save_path + '/' + str(index) + '/clinical_cat',clinical_cat)
        np.save(save_path + '/' + str(index) + '/labels_time',labels_time)
        np.save(save_path + '/' + str(index) + '/labels_censored',labels_censored)
        np.save(save_path + '/' + str(index) + '/ID',ID)
        np.save(save_path + '/' + str(index) + '/mut',mut)
        np.save(save_path + '/' + str(index) + '/cnv_del',cnv_del)
        np.save(save_path + '/' + str(index) + '/cnv_amp',cnv_amp)
        
