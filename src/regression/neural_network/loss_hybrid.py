import torch
import torch.nn as nn

from regression.neural_network.ilr_torch import ilr_transform

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor, reduction: str = "mean"):
        """
        weights: (K,) tensor. Larger => that component matters more.
        reduction: "mean" or "sum" or "none" (per-sample)
        """
        super().__init__()
        self.register_buffer("weights", weights.float())
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds/targets: (B, K)
        w = self.weights.view(1, -1)
        se = (preds - targets) ** 2
        weighted = w * se  # (B, K)

        if self.reduction == "none":
            return weighted.sum(dim=1)  # (B,)
        if self.reduction == "sum":
            return weighted.sum()
        # mean: mean over batch and components
        return weighted.mean()

class WeightedL1Loss(nn.Module):
    def __init__(self, weights: torch.Tensor, reduction: str = "mean"):
        """
        weights: (K,) tensor. Larger => that component matters more.
        reduction: "mean" or "sum" or "none" (per-sample)
        """
        super().__init__()
        self.register_buffer("weights", weights.float())
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds/targets: (B, K)
        w = self.weights.view(1, -1)
        se = torch.abs(preds - targets)  # (B, K)
        weighted = w * se  # (B, K)

        if self.reduction == "none":
            return weighted.sum(dim=1)  # (B,)
        if self.reduction == "sum":
            return weighted.sum()
        # mean: mean over batch and components
        return weighted.mean()
    

class WeightedILRMSELoss(nn.Module):
    """Weighted MSE in ILR-transformed space"""
    def __init__(self, weights: torch.Tensor, reduction: str = "mean"):
        """
        weights: (K,) tensor. Larger => that component matters more.
        reduction: "mean" or "sum" or "none" (per-sample)
        """
        super().__init__()
        self.register_buffer("weights", weights.float())
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction
        self.weights = weights
    
    def forward(self, pred_props: torch.Tensor, target_props: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_props: (batch, n_cell_types) softmax output
            target_props: (batch, n_cell_types) ground truth
        """
        # Transform to ILR space
        pred_ilr = ilr_transform(pred_props, handle_zeros = True)      # (batch, n_cell_types - 1)
        target_ilr = ilr_transform(target_props, handle_zeros = True)  # (batch, n_cell_types - 1)
        # weights = ilr_transform(self.weights.unsqueeze(0))  # (1, n_cell_types - 1)

        # MSE in ILR space
        # weighted = weights * (pred_ilr - target_ilr) ** 2  # (batch, n_cell_types - 1)
        weighted = (pred_ilr - target_ilr) ** 2  # (batch, n_cell_types - 1)

        if self.reduction == "none":
            return weighted.sum(dim=1)  # (B,)
        if self.reduction == "sum":
            return weighted.sum()
        # mean: mean over batch and components
        return weighted.mean()
    

class WeightedILRMSE_Focal_CorrLoss(nn.Module):
    """Weighted ILR-MSE + Focal + Per-cell-type Correlation"""
    def __init__(
        self, 
        cell_type_weights: torch.Tensor,
        gamma: float = 2.0,
        w_ilr: float = 0.5,
        w_focal: float = 0.25,
        w_corr: float = 0.25,
        reduction: str = "mean"
    ):
        super().__init__()
        self.cell_type_weights = cell_type_weights
        self.gamma = gamma
        self.w_ilr = w_ilr
        self.w_focal = w_focal
        self.w_corr = w_corr
        self.reduction = reduction
    
    def forward(self, pred_props, target_props):
        total = 0.0
        if self.w_ilr != 0:
            # Component 1: Weighted ILR-MSE
            ilr_loss = self.ilr_mse(pred_props, target_props)
            if torch.isnan(ilr_loss) or torch.isinf(ilr_loss):
                raise RuntimeError(f"NaN/Inf in ilr_loss: {ilr_loss}")
            total = total + self.w_ilr * ilr_loss
        
        if self.w_focal != 0:
            # Component 2: Focal regression
            focal_loss = self.focal_regression(pred_props, target_props)
            if torch.isnan(focal_loss) or torch.isinf(focal_loss):
                raise RuntimeError(f"NaN/Inf in ilr_loss: {focal_loss}")
            total = total + self.w_focal * focal_loss

        if self.w_corr != 0:
            # Component 3: Per-cell-type correlation
            corr_loss = self.celltype_correlation_loss(pred_props, target_props)
            if torch.isnan(corr_loss) or torch.isinf(corr_loss):
                raise RuntimeError(f"NaN/Inf in ilr_loss: {corr_loss}")
            total = total + self.w_corr * corr_loss

        if torch.isnan(total) or torch.isinf(total):
            raise RuntimeError(f"NaN/Inf in total loss: {total}")

        return total
    
    def ilr_mse(self, pred_props: torch.Tensor, target_props: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Args:
            pred_props: (batch, n_cell_types) softmax output
            target_props: (batch, n_cell_types) ground truth
        """ 
        # Transform to ILR space
        pred_props = pred_props.clamp(min=eps)
        pred_props = pred_props / pred_props.sum(dim=1, keepdim=True).clamp_min(eps)

        target_props = target_props.clamp(min=eps)
        target_props = target_props / target_props.sum(dim=1, keepdim=True).clamp_min(eps)

        pred_ilr = ilr_transform(pred_props, handle_zeros = True)      # (batch, n_cell_types - 1)
        target_ilr = ilr_transform(target_props, handle_zeros = True)  # (batch, n_cell_types - 1)
        # weights = ilr_transform(self.weights.unsqueeze(0))  # (1, n_cell_types - 1)

        # MSE in ILR space
        # weighted = weights * (pred_ilr - target_ilr) ** 2  # (batch, n_cell_types - 1)
        weighted = (pred_ilr - target_ilr) ** 2  # (batch, n_cell_types - 1)

        if self.reduction == "none":
            return weighted.sum(dim=1)  # (B,)
        if self.reduction == "sum":
            return weighted.sum()
        # mean: mean over batch and components
        return weighted.mean()
    
    def focal_regression(self, pred_props, target_props):
        """From Step 2"""
        abs_error = torch.abs(pred_props - target_props)
        # focal_weight = (1 + abs_error) ** self.gamma
        focal_weight =  abs_error**self.gamma
        squared_error = (pred_props - target_props) ** 2
        return (focal_weight * squared_error).mean()
    
    def celltype_correlation_loss(self, pred, target):
        """Per-cell-type Pearson correlation"""
        n_cell_types = pred.shape[1]
        correlations = []
        
        for i in range(n_cell_types):
            corr = self.pearson_correlation(pred[:, i], target[:, i])
            correlations.append(corr)
        
        mean_corr = torch.stack(correlations).mean()
        return 1 - mean_corr  # Convert to loss
    
    def pearson_correlation(self, x, y):
        """Pearson correlation"""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
        return numerator / (denominator + 1e-8)
    


