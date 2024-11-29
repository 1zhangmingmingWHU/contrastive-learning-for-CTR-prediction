import torch
from torch.optim.optimizer import Optimizer
from models.multi_task import MMOE, SharedBottom, PLE, AITM


def shared_task_layers(model):
    """get shared layers and task layers in multi-task model
    Authors: Qida Dong, dongjidan@126.com

    Args:
        model (torch.nn.Module): only support `[MMOE, SharedBottom, PLE, AITM]`

    Returns:
        list[torch.nn.parameter]: parameters split to shared list and task list.
    """
    shared_layers = list(model.embedding.parameters())
    task_layers = None
    if isinstance(model, SharedBottom):
        shared_layers += list(model.bottom_mlp.parameters())
        task_layers = list(model.towers.parameters()) + list(model.predict_layers.parameters())
    elif isinstance(model, MMOE):
        shared_layers += list(model.experts.parameters())
        task_layers = list(model.towers.parameters()) + list(model.predict_layers.parameters())
        task_layers += list(model.gates.parameters())
    elif isinstance(model, PLE):
        shared_layers += list(model.cgc_layers.parameters())
        task_layers = list(model.towers.parameters()) + list(model.predict_layers.parameters())
    elif isinstance(model, AITM):
        shared_layers += list(model.bottoms.parameters())
        task_layers = list(model.info_gates.parameters()) + list(model.towers.parameters()) + list(
            model.aits.parameters())
    else:
        raise ValueError(f'this model {model} is not suitable for MetaBalance Optimizer')
    return shared_layers, task_layers


class MetaBalance(Optimizer):
    """MetaBalance Optimizer
    This method is used to scale the gradient and balance the gradient of each task.
    Authors: Qida Dong, dongjidan@126.com

    Args:
        parameters (list): the parameters of model
        relax_factor (float, optional): the relax factor of gradient scaling (default: 0.7)
        beta (float, optional): the coefficient of moving average (default: 0.9)
    """

    def __init__(self, parameters, relax_factor=0.7, beta=0.9):

        if relax_factor < 0. or relax_factor >= 1.:
            raise ValueError(f'Invalid relax_factor: {relax_factor}, it should be 0. <= relax_factor < 1.')
        if beta < 0. or beta >= 1.:
            raise ValueError(f'Invalid beta: {beta}, it should be 0. <= beta < 1.')
        rel_beta_dict = {'relax_factor': relax_factor, 'beta': beta}
        super(MetaBalance, self).__init__(parameters, rel_beta_dict)

    @torch.no_grad()
    def step(self, losses):
        for idx, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for gp in group['params']:
                    if gp.grad is None:
                        # print('breaking')
                        break
                    if gp.grad.is_sparse:
                        raise RuntimeError('MetaBalance does not support sparse gradients')
                    # store the result of moving average
                    state = self.state[gp]
                    if len(state) == 0:
                        for i in range(len(losses)):
                            if i == 0:
                                gp.norms = [0]
                            else:
                                gp.norms.append(0)
                    # calculate the moving average
                    beta = group['beta']
                    gp.norms[idx] = gp.norms[idx] * beta + (1 - beta) * torch.norm(gp.grad)
                    # scale the auxiliary gradient
                    relax_factor = group['relax_factor']
                    gp.grad = gp.grad * gp.norms[0] / (gp.norms[idx] + 1e-5) * relax_factor + gp.grad * (1. -
                                                                                                         relax_factor)
                    # store the gradient of each auxiliary task in state
                    if idx == 0:
                        state['sum_gradient'] = torch.zeros_like(gp.data)
                        state['sum_gradient'] += gp.grad
                    else:
                        state['sum_gradient'] += gp.grad

                    if gp.grad is not None:
                        gp.grad.detach_()
                        gp.grad.zero_()
                    if idx == len(losses) - 1:
                        gp.grad = state['sum_gradient']