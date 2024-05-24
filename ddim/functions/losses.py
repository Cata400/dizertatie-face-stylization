import torch


def noise_estimation_loss(model,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e: torch.Tensor,
                        b: torch.Tensor, 
                        content_vgg=None, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float(), content_vgg)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
    
def noise_estimation_loss_vgg(model,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e: torch.Tensor,
                        b: torch.Tensor, 
                        content_vgg: torch.Tensor=None, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float(), content_vgg)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
    
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

    
def noise_estimation_loss_vgg_id(model,
                        id_loss_func,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e: torch.Tensor,
                        b: torch.Tensor,
                        eta: float, 
                        content_vgg: torch.Tensor=None, 
                        keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    
    et = model(x, t.float(), content_vgg)
    
    t_next = t - 1
    at_next = compute_alpha(b, t_next.long())
    
    x0_t = (x - et * (1 - a).sqrt()) / a.sqrt()
    c1 = eta * ((1 - a / (at_next)) * (1 - at_next) / (1 - a)).sqrt()
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x)
    
    loss_id = torch.mean(id_loss_func(x0, xt_next))
    # loss_l1 = torch.nn.L1Loss()(x0, xt_next)
    loss_l2 = torch.nn.MSELoss()(e, et)
    
    loss = 0.5 * loss_id + 0.5 * loss_l2
    return loss
    



loss_registry = {
    'simple': noise_estimation_loss,
    'vgg': noise_estimation_loss_vgg,
    'vgg_id': noise_estimation_loss_vgg_id,
}
