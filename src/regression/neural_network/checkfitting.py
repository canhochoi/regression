import numpy as np
from skbio.stats.composition import ilr_inv, clr

__all__ = ["CheckFitting"]

class CheckFitting():
    '''
    A class to check the fitting of regression models
    '''
    @staticmethod    
    def get_cor(one_fold_result, transform = "ilr_inv"):
        cor_list = []
        if transform == "ilr_inv":
            for i in np.arange(one_fold_result['Y_pred_pytorch'].shape[1] + 1):
            
                cor_list.append(np.corrcoef(ilr_inv(one_fold_result['Y_pred_pytorch'])[:, i],
                                            ilr_inv(one_fold_result['processed_dict']['Y_test_t'])[:, i],
                                            rowvar = False)[0][1])
        elif transform == "clr":
            # agree well with analytical result
            # Y_pytorch_ct = ilr_inv(one_fold_result['Y_pred_pytorch']) 
            # Y_test_ct = ilr_inv(one_fold_result['processed_dict']['Y_test_t'])
            tr_t_mean = one_fold_result['processed_dict']['tr_t_mean'].cpu().numpy()
            Y_pytorch_ct = ilr_inv(one_fold_result['Y_pred_pytorch'] + tr_t_mean) 
            Y_test_ct = ilr_inv(one_fold_result['processed_dict']['Y_test_t']+ tr_t_mean)
            
            
            for i in np.arange(one_fold_result['Y_pred_pytorch'].shape[1] + 1):
            
                cor_list.append(np.corrcoef(clr(Y_pytorch_ct)[:, i],
                                            clr(Y_test_ct)[:, i],
                                            rowvar = False)[0][1])        
        else:        
            # for ilr-transformed axes only
            for i in np.arange(one_fold_result['Y_pred_pytorch'].shape[1]):
                cor_list.append(np.corrcoef(one_fold_result['Y_pred_pytorch'][:, i], one_fold_result['processed_dict']['Y_test_t'][:, i], rowvar = False)[0][1])
        return cor_list    