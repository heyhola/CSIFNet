import os
import numpy as np 
import ml_collections
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.nn.utils.weight_norm import weight_norm
from .components import OutputBlock, CLEM, FUSION, imagemodel, AutoEncoder, self_Attention
from ban import BANLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':16})
    config.hidden_size = 128
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 8
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = 'token'
    config.representation_size = None
    return config

trans_config=get_config()

class Network(nn.Module):

    def __init__(self, opts, n_out_features, batch_size, device, 
                 n_cont_cols, n_classes_cat, gene, vail_batch_size = 1, vail = False):
        super(Network, self).__init__()
        channels = [opts.initial_depth, opts.initial_depth*2, opts.initial_depth*4, opts.initial_depth*8]
    
        # CNN backbone
        resnet_type = 'resnet50'

        self.fv_dim = 128
        self.n_clinical = 9
        self.n_pixel = 6*6
        self.n_nodes = self.n_clinical + self.n_pixel
        self.genes = gene

        n_aux_classes = [1]*n_cont_cols + n_classes_cat
        self.batch_size = batch_size
        self.vail_batch_size = vail_batch_size
        self.multimodal = np.zeros((1,10,128))
        # Model architecture
        self.image_5 = imagemodel(resnet_type, 5, channels, self.fv_dim)
        self.image_39 = imagemodel(resnet_type, 39, channels, self.fv_dim)
        self.image_50 = imagemodel(resnet_type, 50, channels, self.fv_dim)
        self.SIFM = weight_norm(
            BANLayer(v_dim=128, q_dim=128, h_dim=512, h_out=2),
            name='h_mat', dim=None)
        
        self.clinical_mlps = CLEM(self.fv_dim, self.n_clinical, n_aux_classes)

        self.graph_net = FUSION(self.fv_dim)
        self.output_mlp = OutputBlock(self.fv_dim * (self.n_clinical+2), n_out_features)#raw
        # # Autoencoder
        hidden_layers = [5096, 2048, 1024, 128]
        self.autoencoder = AutoEncoder(self.genes[0] + self.genes[1] + self.genes[2], hidden_layers)
        self.attn = self_Attention(trans_config,vis=True)

    def get_edges(self, n_clinical_omic, n_nodes):        
        node_ids = np.expand_dims(np.arange(n_nodes, dtype=int), 0)
        self_edges = np.concatenate((node_ids, node_ids), 0)
        c_array_asc = np.expand_dims(np.arange(n_clinical_omic), 0)
        all_edges = self_edges[:]

        for i in range(n_clinical_omic, n_nodes):
            i_array = np.expand_dims(np.array([i]*n_clinical_omic), 0)
            inter_edges_ic = np.concatenate((i_array, c_array_asc), 0) 
            inter_edges_ci = np.concatenate((c_array_asc, i_array), 0)
            inter_edges_i = np.concatenate((inter_edges_ic, inter_edges_ci), 1)
            all_edges = np.concatenate((all_edges, inter_edges_i), 1)
        all_edges = torch.tensor(all_edges, dtype=torch.long)
        return all_edges
    
    def forward(self, in_img, in_clinical, in_omics, vail = False):
        #### tumor cell ####
        if in_img.shape[1] == 50:
            conv_1x1_reshape = self.image_50(in_img, vail)
           
        #### microenvironment ####
        elif in_img.shape[1] == 44:
            VesselMask = in_img[:,0:1,:,:]
            NucleusMask = in_img[:,1:2,:,:]
            FullStack_39 = in_img[:,2:41,:,:]
            EpithleialMask = in_img[:,41:42,:,:]
            CytoplasmMask = in_img[:,42:43,:,:]
            CellMask = in_img[:,43:,:,:]
            #### concat ####
            single_channel = torch.cat((VesselMask, NucleusMask, EpithleialMask, CytoplasmMask, CellMask),1)
            conv_1x1_reshape_1 = self.image_5(single_channel, vail)
            conv_1x1_reshape_39 = self.image_39(FullStack_39, vail)
            ###image features concat
            conv_1x1_reshape = torch.cat((conv_1x1_reshape_1,conv_1x1_reshape_39),axis=1)
        #### ALL ####
        # imagetype = [VesselMask, NucleusMask, FullStack(39), EpithleialMask, CytoplasmMask, CellMask, fullstack(50)]
        elif in_img.shape[1] == 94:
            VesselMask = in_img[:,0:1,:,:]
            NucleusMask = in_img[:,1:2,:,:]
            FullStack_39 = in_img[:,2:41,:,:]
            EpithleialMask = in_img[:,41:42,:,:]
            CytoplasmMask = in_img[:,42:43,:,:]
            CellMask = in_img[:,43:44,:,:]
            fullstack_50 = in_img[:,44:,:,:]
            single_channel = torch.cat((VesselMask, NucleusMask, EpithleialMask, CytoplasmMask, CellMask),1)
            conv_1x1_reshape_1 = self.image_5(single_channel, vail)
            conv_1x1_reshape_39 = self.image_39(FullStack_39, vail)
            conv_1x1_reshape_50 = self.image_50(fullstack_50, vail)

            conv_1x1_reshape_mirco = torch.cat((conv_1x1_reshape_1,conv_1x1_reshape_39),axis=1)
            f2, att_final = self.SIFM(conv_1x1_reshape_mirco, conv_1x1_reshape_50)
           
            self.att_final = att_final
            conv_1x1_reshape = f2.reshape(1,-1,128) if vail else f2.reshape(8,-1,128)
           
        # Extract clinical features [batch_size, n_clinical, fv_dim]
        clinical_fvs, clinical_preds = self.clinical_mlps(in_clinical)
        # AE
        encoded, decoded = self.autoencoder(in_omics)
        omics_fvs = encoded.unsqueeze(0).unsqueeze(0) if vail else encoded.unsqueeze(1)
        omics_out = decoded
        clinical_omics_fvs = torch.cat((clinical_fvs, omics_fvs),1)
        if vail:
            self.multimodal = np.concatenate((self.multimodal,clinical_omics_fvs.cpu()))
        clinical_fvs,_ = self.attn(clinical_omics_fvs)
       
        self.n_clinical = 10
        if in_img.shape[1] == 50:
            self.n_pixel = 6*6
        elif in_img.shape[1] == 44:
            self.n_pixel = 6*6*2
        elif in_img.shape[1] == 94:
            self.n_pixel = int(self.SIFM.h_dim / self.fv_dim)
           
        self.n_nodes = self.n_clinical + self.n_pixel    
        self.edge_index = (self.get_edges(self.n_clinical, self.n_nodes)).to(device)     
        for ind in range(self.vail_batch_size) if vail else range(self.batch_size):
            if ind == 0:
                batch_semantic_fvs = clinical_fvs[0,:,:]
                batch_semantic_fvs = torch.cat((batch_semantic_fvs, conv_1x1_reshape[0,:,:]),0)
            else:
                batch_semantic_fvs = torch.cat((batch_semantic_fvs, clinical_fvs[ind,:,:]),0)
                batch_semantic_fvs = torch.cat((batch_semantic_fvs, conv_1x1_reshape[ind,:,:]),0)
                

        batch_edge_index = self.edge_index.clone()
        for ind in range(1,self.vail_batch_size) if vail else range(1, self.batch_size):
            next_edge_index = self.edge_index + self.n_nodes*ind
            batch_edge_index = torch.cat((batch_edge_index, next_edge_index), 1)

        data = Data(x=batch_semantic_fvs, edge_index=batch_edge_index)

        # # GAT
        batch_graph_fvs = self.graph_net(data)

        for ind in range(self.vail_batch_size) if vail else range(self.batch_size):
            if ind == 0:
                graph_fvs_c = batch_graph_fvs[:self.n_clinical,:].unsqueeze(0)
                graph_fvs_i = batch_graph_fvs[self.n_clinical:self.n_clinical+self.n_pixel,:].unsqueeze(0)
            else:
                graph_fvs_c = torch.cat((graph_fvs_c,
                                            batch_graph_fvs[ind*self.n_nodes:ind*self.n_nodes+self.n_clinical,:].unsqueeze(0)),
                                            0)
                graph_fvs_i = torch.cat((graph_fvs_i,
                                            batch_graph_fvs[ind*self.n_nodes+self.n_clinical:ind*self.n_nodes+self.n_clinical+self.n_pixel,:].unsqueeze(0)),
                                            0)                

        # [batch_size, fv_dim, n_pixel]
        graph_fvs_i = torch.transpose(graph_fvs_i, 1, 2)
        # Vectorise
        if in_img.shape[1] == 50:
            self.gap = nn.AvgPool2d(kernel_size=(6,6))
            gap = self.gap(torch.reshape(graph_fvs_i, (self.vail_batch_size if vail else self.batch_size, self.fv_dim, 6, 6))).squeeze(-1).squeeze(-1)
        elif in_img.shape[1] == 44:
            self.gap = nn.AvgPool2d(kernel_size=(6,6*2))
            gap = self.gap(torch.reshape(graph_fvs_i, (self.vail_batch_size if vail else self.batch_size, self.fv_dim, 6, 6*2))).squeeze(-1).squeeze(-1)
        elif in_img.shape[1] == 94:
            self.gap = nn.AvgPool2d(kernel_size=(2,2))
            gap = self.gap(torch.reshape(graph_fvs_i, (self.vail_batch_size if vail else self.batch_size, self.fv_dim, 2, 2))).squeeze(-1).squeeze(-1)
        combined = torch.cat((graph_fvs_c, gap.unsqueeze(1)), 1)
        # reshape
        combined = torch.reshape(combined, (self.vail_batch_size if vail else self.batch_size, -1))

        # Get predictions
        feature_preds = self.output_mlp(combined)
        
        return feature_preds, clinical_preds, omics_out
