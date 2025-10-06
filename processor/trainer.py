from gudhi.wasserstein import wasserstein_distance
import torch
import torch.nn.functional as F
from processor.graph import sparse_to_tuple
from processor.model import GAEMF
import time
from processor.filtration import WrcfLayer, build_community_graph


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.long()
    preds_all = torch.gt(adj_rec, 0.5).long()
    return torch.eq(labels_all,preds_all).float().mean()


class TopoLoss(torch.nn.Module):

    def __init__(self, nearby_dgms, args) -> None:
        super().__init__()
        self.wrcf_layer_dim0 = WrcfLayer(dim=0, card=args.card)
        self.wrcf_layer_dim1 = WrcfLayer(dim=1, card=args.card)
        self.dgm_gt = nearby_dgms
        self.LAMBDA = args.LAMBDA

    def forward(self, adj, soft_label):
        target_device = soft_label.device
        # compute graph filtration based topological loss
        C = build_community_graph(soft_label, adj)
        dgm_dim0 = self.wrcf_layer_dim0(C)
        dgm_dim1 = self.wrcf_layer_dim1(C)

        # loss_topo = torch.square(wasserstein_distance(dgm, self.dgm_gt, order=2,
        #                 enable_autodiff=True, keep_essential_parts=False))
        if self.dgm_gt[0]:
            topo_dim0_before = wasserstein_distance(dgm_dim0,
                                            self.dgm_gt[0][0],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
            topo_dim1_before = wasserstein_distance(dgm_dim1,
                                            self.dgm_gt[0][1],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
        if self.dgm_gt[1]:
            topo_dim0_next = wasserstein_distance(dgm_dim0,
                                            self.dgm_gt[1][0],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
            

            topo_dim1_next = wasserstein_distance(dgm_dim1,
                                            self.dgm_gt[1][1],
                                            order=1,
                                            # internal_p=2,
                                            enable_autodiff=True,
                                            keep_essential_parts=False)
        
        if not self.dgm_gt[0]:
            loss_topo = topo_dim0_next + topo_dim1_next
        elif not self.dgm_gt[1]:
            loss_topo = topo_dim0_before + topo_dim1_before
        else:
            loss_topo = topo_dim0_before + topo_dim1_before + topo_dim0_next + topo_dim1_next
        return  self.LAMBDA * loss_topo.to(target_device)
    
    
def gaemf_trainer(
        model:GAEMF,
        features:torch.Tensor,
        adj:torch.Tensor,
        args:dict, 
        topo:TopoLoss,
        idx:str
    ):
    # Use the same device as the model parameters to avoid device mismatch
    device = next(model.parameters()).device
    adj_label = sparse_to_tuple(adj)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                        torch.FloatTensor(adj_label[1]),
                                        torch.Size(adj_label[2])).to(device)


    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device)
    weight_tensor[weight_mask] = pos_weight

    if topo:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate,
                                  weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate,
                                  weight_decay=1e-4)
    model.train()
    for epoch in range(args.num_epoch):
        t = time.time()
        mf_flag = epoch>args.start_mf
        if topo:
            mf_flag = True
        A_pred, z, q = model(features, mf_flag)
        re_loss = norm * F.binary_cross_entropy(A_pred.view(-1),
                                                 adj_label.to_dense().view(-1),
                                                 weight=weight_tensor)

        if topo:
            loss = topo(adj,q) + re_loss
        elif epoch>args.start_mf:
            loss_kmeans = F.mse_loss(z, torch.mm(q, model.cluster_centroid))
            loss = 1 * loss_kmeans + re_loss
        else:
            loss = re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            if int(idx)>=9 and args.file_name == "Data/Cora":
                train_acc = 0
            else:
                train_acc = get_acc(A_pred, adj_label.to_dense())
            print("Epoch:", '%04d' % (epoch + 1), 
              "extra_loss=", "{:.5f}".format(loss.item() - re_loss.item()),
              "re_loss=", "{:.5f}".format(re_loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), 
              "time=", "{:.5f}".format(time.time() - t))
            

def base_train(model,features,adj,args,idx):
    gaemf_trainer(model,features,adj,args,None,idx)
            

def retrain_with_topo(_model, dgm_gt, adj, features, args, idx):
    topo_loss = TopoLoss(nearby_dgms=dgm_gt, args=args)
    # train model
    gaemf_trainer(_model,features,adj,args,topo_loss,idx)
