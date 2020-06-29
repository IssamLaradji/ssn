import numpy as np
import torch
import ssn


def get_optimizer(opt_dict, params, n_batches_per_epoch=None, n_train=None,
                  train_loader=None, exp_dict=None, loss_function=None, model=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    opt_name = opt_dict["name"]

    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch", n_batches_per_epoch) 
    
    if opt_name == 'ssn':
        opt = ssn.Ssn(params, 
            n_batches_per_epoch=n_batches_per_epoch, 
            init_step_size=1.0, 
            lr=None, 
            c=0.1, 
            beta=0.9, 
            gamma=1.5,
            reset_option=1, 
            lm=opt_dict.get("lm", 0))

    elif opt_name == "sgd_armijo":
        c = opt_dict.get("c", 0.1)
        
        opt = ssn.Sls(params,
                    c = c,
                    n_batches_per_epoch=n_batches_per_epoch,
                    line_search_fn="armijo", 
                    gamma=opt_dict.get("gamma", 2.0),
                    reset_option=opt_dict.get("reset_option", 1),
                    eta_max=opt_dict.get("eta_max"))

    elif opt_name == "sgd_polyak":
        opt = ssn.SlsAcc(params, 
                         c=opt_dict.get("c") or 0.1,
                         momentum=opt_dict.get("momentum", 0.6),
                         n_batches_per_epoch=n_batches_per_epoch,
                         gamma=opt_dict.get("gamma", 2.0),
                         acceleration_method="polyak",
                         eta_max=opt_dict.get("eta_max"),
                         reset_option=opt_dict.get("reset", 1))

    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(params, 
                                lr=0.9, 
                                max_iter=opt_dict.get("max_iter", 20),
                                history_size=opt_dict.get("history_size", 100),
                                line_search_fn="strong_wolfe")

    elif opt_name == "svrg":
        lr = get_svrg_step_size(exp_dict)
        n = len(train_loader.dataset)
        full_grad_closure = svrg.full_loss_closure_factory(train_loader,
                                                        loss_function,
                                                        grad=True)
        opt = svrg.SVRG(model,
                        train_loader.batch_size,
                        lr,
                        n,
                        full_grad_closure,
                        m=len(train_loader),
                        splr_flag=exp_dict['opt'].get('splr_flag'),
                        c=exp_dict['opt'].get('c'),
                        
                        )

    elif opt_name == "slbfgs":
        opt = slbfgs.LBFGS(params, 
                                lr=opt_dict.get("lr"),
                                history_size=opt_dict.get("history_size", 100),
                                max_iter=2,
                                line_search_fn=opt_dict.get("line_search_fn"),
                                lm=opt_dict.get("lm", 0)
                                )

    elif opt_name == "adam":
        best_lr = opt_dict.get('lr', 0.001)
        opt = torch.optim.Adam(params, lr=best_lr)

    elif opt_name == "adagrad":
        best_lr = opt_dict.get('lr', 0.01)
        opt = torch.optim.Adagrad(params, lr=best_lr)

    elif opt_name == 'sgd':
        # best_lr = lr if lr else 1e-3
        opt = torch.optim.SGD(params, lr=opt['lr'])

    elif opt_name == "sgd-m":
        best_lr = lr if lr else 1e-3
        opt = torch.optim.SGD(params, lr=best_lr, momentum=0.9)

    elif opt_name == 'rms':
        opt = torch.optim.RMSprop(params)

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt



# learning rates selected by cross-validation.
lr_dict = {
    "logistic_loss": {          
                    "rcv1"      : 500,
                    "mushrooms" : 500,
                    "ijcnn"     : 500,
                    "w8a"       : 0.0025,
                    'syn-0.01'  : 1.5,
                    'syn-0.05'  : 0.1,
                    'syn-0.1'   : 0.025,
                    'syn-0.5'   : 0.0025,
                    'syn-1.0'   : 0.001,},
    "squared_hinge_loss" : {            
                    'mushrooms' : 150., 
                    'rcv1'      : 3.25,
                    "ijcnn"     : 2.75,
                    "w8a"       : 0.00001,
                    'syn-0.01'  : 1.25,
                    'syn-0.05'  : 0.025,
                    'syn-0.1'   : 0.0025,
                    'syn-0.5'   : 0.001,
                    'syn-1.0'   : 0.001,}
}

def get_svrg_step_size(exp_dict):
    ds_name = exp_dict["dataset"]
    if ds_name == "synthetic":
        ds_name = "syn-%s" % str(exp_dict["margin"])
    lr = lr_dict[exp_dict["loss_func"]][ds_name]

    return lr