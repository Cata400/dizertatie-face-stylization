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


loss_registry = {
    'simple': noise_estimation_loss,
    'vgg': noise_estimation_loss_vgg,
}