import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import logging
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lifelines.statistics import multivariate_logrank_test
from dataio.data_process import DatasetLIDC
from model.model import Network
from utils.utils import *
from lifelines import KaplanMeierFitter
from omics.Reactome import *
import numpy as np
from plot import draw_CAM
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import random
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(config):
    config_fold = config.config_file + str(2) + '.json'
    json_opts = json_file_to_pyobj(config_fold)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # Create experiment directories
    if config.resume_epoch == None:
        make_new = True 
    else:
        make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments_orignal' + '/' + timestamp
    make_dir(experiment_path + '/' + json_opts.experiment_dirs.model_dir)
    image_path = 'out_image' + '/' + timestamp
    make_dir(image_path)
   
    fold_mean = json_opts.data_params.fold_means
    fold_std = json_opts.data_params.fold_stds
    fold_mean_full = json_opts.data_params.fold_means_full
    fold_std_full = json_opts.data_params.fold_std_full
    fold_mean_other = json_opts.data_params.fold_means_other
    fold_std_other = json_opts.data_params.fold_std_other
    
    mut = list(np.load(json_opts.omic_source.mut_file))
    cnv_del = list(np.load(json_opts.omic_source.cnv_del_file))
    cnv_amp = list(np.load(json_opts.omic_source.cnv_amp_file))
    all_genes = [len(mut), len(cnv_del) ,len(cnv_amp)]
    assert(len(fold_mean) == len(fold_std))

    # Set up the model
    logging.info("Initialising model")
    model_opts = json_opts.model_params
    n_out_features = 1
    n_classes = [1]*n_out_features

    # Input variables and class mappings for categorical variables
    # cont_cols = ['Age.At.Diagnosis', 'Ki67', 'EGFR', 'PR', 'HER2']
    cont_cols = ['Age.At.Diagnosis', 'Ki67', 'EGFR', 'PGR', 'ERBB2']
    cat_cols = ['CT', 'RT', 'HT', 'ER.Status']

    di_ct = {"NO/NA": 0, "ECMF": 1, "OTHER": 1, "AC": 1, "CAPE": 1, 
             "AC/CMF": 1, "CMF": 1, "PACL": 1, "FAC": 1}    
    di_rt = {"CW": 1, "NO/NA": 0, "CW-NODAL": 1, "NONE RECORDED IN LANTIS": 0}    
    di_ht = {"TAM": 1, "NO/NA": 0, "TAM/AI": 1, "AI": 1, "GNRHA": 1, 
             "OO": 1, "OTHER": 1, "Y": 1}   
    di_er = {"neg": 0, "pos": 1}    
    n_classes_cat = [2, 2, 2, 2] 

    model = Network(model_opts, n_out_features,
                    json_opts.training_params.batch_size, device,
                    len(cont_cols), n_classes_cat, all_genes)
    model = model.to(device)  
   
    # Dataloader
    logging.info("Preparing data")
    num_workers = json_opts.data_params.num_workers
    train_dataset = DatasetLIDC(json_opts.data_source, config.fold, fold_mean, fold_std,
                                json_opts.data_params.in_h, json_opts.data_params.in_w,
                                fold_mean_full, fold_std_full, fold_mean_other, fold_std_other,
                                di_ct, di_rt, di_ht, di_er, cont_cols, cat_cols, n_classes_cat, 
                                isTraining=True)
   
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=json_opts.training_params.batch_size, 
                              shuffle=True, num_workers=num_workers, drop_last=True)   
    
    test_dataset = DatasetLIDC(json_opts.data_source, config.fold, fold_mean, fold_std,
                                json_opts.data_params.in_h, json_opts.data_params.in_w,
                                fold_mean_full, fold_std_full, fold_mean_other, fold_std_other,
                                di_ct, di_rt, di_ht, di_er, cont_cols, cat_cols, n_classes_cat,
                                isTraining=False)
    test_loader = DataLoader(dataset=test_dataset, 
                              batch_size=1, 
                              shuffle=False, num_workers=num_workers)

    n_test_examples = len(test_loader)
    # n_train_examples = len(train_loader.dataset)

    # Auxiliary losses and optimiser
    criterion_mae = torch.nn.MSELoss(reduction='sum')
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam([
        {'params': model.image_5.parameters(),},
        {'params': model.image_39.parameters(),},
        {'params': model.image_50.parameters(),},
        {'params': model.clinical_mlps.parameters(),},
        {'params': model.graph_net.parameters(),},
        {'params': model.output_mlp.parameters(),},
        {'params': model.SIFM.parameters(),},
        {'params': model.attn.parameters(),'lr':0.0001},
        {'params': model.autoencoder.parameters(),'lr':0.1},],
        lr=json_opts.training_params.learning_rate, 
                                 betas=(json_opts.training_params.beta1, 
                                        json_opts.training_params.beta2),
                                 weight_decay=json_opts.training_params.l2_reg_alpha)
    
    if config.resume_epoch != None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    if config.resume_epoch != None:
        load_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                    "/epoch_%d.pth" %(config.resume_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == config.resume_epoch)
        print("Resume training, successfully loaded " + load_path)

    logging.info("Begin training")

    model = model.train()

    best_result = 0    

    for epoch in range(initial_epoch, json_opts.training_params.total_epochs):
        epoch_train_loss = 0.
        model.train() 
    
        for _, (batch_x, batch_c, batch_cat, batch_y, death_indicator, batch_mut, batch_cnv_del, batch_cnv_amp) in enumerate(train_loader):

            # Transfer to GPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_c, batch_cat = batch_c.to(device), batch_cat.to(device)
            death_indicator = death_indicator.to(device)
            batch_mut = batch_mut.squeeze().to(device)
            batch_cnv_del = batch_cnv_del.squeeze().to(device)
            batch_cnv_amp = batch_cnv_amp.squeeze().to(device)
            batch_all_gene = torch.cat((batch_mut,batch_cnv_del,batch_cnv_amp),1)
            optimizer.zero_grad()
            # Forward pass
            final_pred, aux_pred, omics_out = model(batch_x, batch_c, batch_all_gene)

            # Optimisation
            if torch.sum(death_indicator) > 0.0:
                losses = []
                
                for lx in range(n_out_features):
                    end_idx = int(np.sum(n_classes[:lx+1]))
                    start_idx = int(end_idx - n_classes[lx])
                    losses.append(PartialLogLikelihood(final_pred[:,start_idx:end_idx], 
                                                    death_indicator, 'noties'))
                
                loss = sum(losses)[0]

                aux_loss = criterion_mae(aux_pred[:,:len(cont_cols)], 
                                        batch_c[:,:len(cont_cols)]).squeeze()
                loss_func = torch.nn.BCEWithLogitsLoss().to(device)
                omics_loss = loss_func(omics_out, batch_all_gene.float())
                loss += json_opts.training_params.lambda_aux*aux_loss 
                
                for lx in range(len(n_classes_cat)):
                    end_idx = int(np.sum(n_classes_cat[:lx+1])) + len(cont_cols)
                    start_idx = int(end_idx - n_classes_cat[lx])
                    aux_loss = criterion_ce(aux_pred[:,start_idx:end_idx], batch_cat[:,lx]).squeeze()
                    loss += json_opts.training_params.lambda_aux*aux_loss 
                    

                loss += json_opts.training_params.lambda_omics_aux * omics_loss
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.detach().cpu().numpy()
            
        logging.info('Epoch[{}/{}], total loss:{:.4f}'.format(epoch + 1, json_opts.training_params.total_epochs, 
                                                      epoch_train_loss))
        
        test_y_true = np.zeros(n_test_examples)
        test_y_out = np.zeros(n_test_examples)
        test_death_true = np.zeros(n_test_examples)
        patient = []
        model.eval() 
        with torch.no_grad():
            for batch_idx, (batch_x, batch_c, batch_cat, batch_y, death_indicator, ID, batch_mut, batch_cnv_del, batch_cnv_amp) in enumerate(test_loader):

                # Transfer to GPU
                batch_x, batch_c, batch_cat, batch_y = batch_x.to(device), batch_c.to(device), batch_cat.to(device), batch_y.to(device)
                death_indicator = death_indicator.to(device)
                batch_mut = batch_mut.squeeze().to(device)
                batch_cnv_del = batch_cnv_del.squeeze().to(device)
                batch_cnv_amp = batch_cnv_amp.squeeze().to(device)
                patient.append(ID)
                batch_all_gene = torch.cat((batch_mut,batch_cnv_del,batch_cnv_amp),0)
                final_pred, _ , _  = model(batch_x, batch_c, batch_all_gene, vail = True)
                # Labels, predictions per example
                test_y_true[batch_idx] = batch_y.squeeze().detach().cpu().numpy()
                test_y_out[batch_idx] = final_pred.squeeze().detach().cpu().numpy()
                test_death_true[batch_idx] = death_indicator.squeeze().detach().cpu().numpy()
   
            # Compute performance
            test_temp = calc_concordance_index(test_y_out, test_death_true, test_y_true)   
            print('test_C-index: mean ', np.around(test_temp,5)) 
            
            if best_result < test_temp:
                best_result = test_temp
                best_model = model
                best_epoch = epoch

    save_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                "/epoch_%d.pth" %(best_epoch+1)
    torch.save({'epoch': best_epoch + 1,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
    logging.info("Model saved: %s" % save_path)
    print('test best C-index: mean ', np.around(best_result,5))
       
    logging.info("Testing finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='resume training from this epoch, set to None for new training')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')
    parser.add_argument('--fold', default=0, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    main(config)

