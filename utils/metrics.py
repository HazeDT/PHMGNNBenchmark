from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
import torch
import torch.nn.functional as F
import models
import numpy as np



#
def RUL_Score(y_true, y_pre):

    y_true = y_true.view(-1).cpu().detach().numpy()
    y_pre = y_pre.view(-1).cpu().detach().numpy()
    d = y_pre - y_true
    mse = np.mean(np.square(d))
    phm_score = np.sum(np.exp(-d[d < 0] / 13) - 1) + np.sum(np.exp(d[d >= 0] / 10) - 1)

    return mse, phm_score




if __name__ == '__main__':
    from datasets.Environment import Environment
    dataset = Environment
    file_root = 'D:/Competitions/人工智能创新应用大赛-智慧环保/Trainset/'
    train, val = dataset(file_root).data_preprare()
    dataloaders = torch.utils.data.DataLoader(val, batch_size=8,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True
                                              )
    model = getattr(models, 'resnet18')(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to('cuda')
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        logits = model(inputs)
        if batch_idx == 0:
            pred_all, labels_all = RocAucEvaluation(logits, labels)
        else:
            pred_tmp, labels_tmp = RocAucEvaluation(logits, labels)
            pred_all = np.concatenate((pred_all, pred_tmp))
            labels_all = np.concatenate((labels_all, labels_tmp))

    print(RocAucEvaluation(pred_all, labels_all, test=True))

