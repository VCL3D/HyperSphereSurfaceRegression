import torch
import torch.nn.functional as F
import numpy as np

# image gradient computations
'''
    Image gradient x-direction
    \param
        input_tensor
    \return 
        input_tensor's x-direction gradients
'''
def grad_x(input_tensor):
    input_tensor = F.pad(input_tensor, (0, 1, 0, 0), mode = "replicate")
    gx = input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:]
    return gx

'''
    Image gradient y-direction
    \param
        input_tensor
    \return 
        input_tensor's y-direction gradients
'''
def grad_y(input_tensor):
    input_tensor = F.pad(input_tensor, (0, 0, 0, 1), mode = "replicate")
    gy = input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :]
    return gy

'''
    L2 Loss
    \param
        input       input tensor (model's prediction)
        target      target tensor (ground truth)
        use_mask    set True to compute masked loss
        mask        Binary mask tensor
    \return
        L2 loss mean between target and input
        L2 loss map between target and input
'''
def l2_loss(input, target, use_mask = True, mask = None):
    loss = torch.pow(target - input, 2)
    if use_mask and mask is not None:
        count = torch.sum(mask).item()
        masked_loss = loss * mask
        return torch.sum(masked_loss) / count, masked_loss
    return torch.mean(loss), loss

'''
    Cosine Similarity loss (vector dot product)
    \param
        input       input tensor (model's prediction)
        target      target tensor (ground truth)
        use_mask    set True to compute masked loss
        mask        Binary mask tensor
    \return
        Cosine similarity loss mean between target and input
        Cosine similarity loss map betweem target and input
'''
def cosine_loss(input, target, use_mask = True, mask = None):
    loss = 2 - (1 + torch.sum(input * target, dim = 1, keepdim = True))
    if use_mask and mask is not None:
        count = torch.sum(mask)
        masked_loss = loss * mask
        return torch.sum(masked_loss) / count, masked_loss
    return torch.mean(loss), loss
'''
    Quaternion loss
    \param
        input       input tensor (model's prediction)
        target      target tensor (ground truth)
        use_mask    set True to compute masked loss
        mask        Binary mask tensor
    \return
        Quaternion loss mean between target and input
        Quaternion loss map betweem target and input
'''
def quaternion_loss(input, target, use_mask = True, mask = None):
    q_pred = -input
    loss_x = target[:, 1, :, :] * q_pred[:, 2, :, :] - target[:, 2, :, :] * q_pred[:, 1, :, :]
    loss_y = target[:, 2, :, :] * q_pred[:, 0, :, :] - target[:, 0, :, :] * q_pred[:, 2, :, :]
    loss_z = target[:, 0, :, :] * q_pred[:, 1, :, :] - target[:, 1, :, :] * q_pred[:, 0, :, :]
    loss_re = -target[:, 0, :, :] * q_pred[:, 0, :, :] - target[:, 1, :, :] * q_pred[:, 1, :, :] - target[:, 2, :, :] * q_pred[:, 2, :, :]
    loss_x = loss_x.unsqueeze(1)
    loss_y = loss_y.unsqueeze(1)
    loss_z = loss_z.unsqueeze(1)
    loss_xyz = torch.cat((loss_x, loss_y, loss_z), 1)
    
    dot = loss_x * loss_x + loss_y * loss_y + loss_z * loss_z
    eps = torch.ones_like(dot) * 1e-8

    vec_diff = torch.sqrt(torch.max(dot, eps))
    real_diff = torch.sign(loss_re) * torch.abs(loss_re)
    real_diff = real_diff.unsqueeze(1)
    
    loss = torch.atan2(vec_diff, real_diff) / (np.pi)

    if mask is not None:
        count = torch.sum(mask)
        mask = mask[:, 0, :, :].unsqueeze(1)
        masked_loss = loss * mask
        return torch.sum(masked_loss) / count, masked_loss
    return torch.mean(loss)

'''
    Smoothness loss
    \param
        input   input tensor (model's prediction)
'''
def smoothness_loss(input, use_mask = True, mask = None):
    grads_x = grad_x(input)
    grads_y = grad_y(input)
    loss = torch.abs(grads_x) + torch.abs(grads_y)
    if mask is not None:
        count = torch.sum(mask).item()
        masked_loss = mask * loss
        return torch.sum(masked_loss) / count, masked_loss
    return torch.mean(loss), loss