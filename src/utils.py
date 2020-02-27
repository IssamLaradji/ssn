import torch 

import numpy as np


def update_trainloader_and_opt(train_set, opt, batch_size, n_train, batch_grow_factor, batch_size_max):
    
    n_iters = (n_train // batch_size)

    batch_size_new = batch_size * batch_grow_factor ** n_iters
    batch_size_new = min(int(batch_size_new), batch_size_max)
    batch_size_new = min(batch_size_new, n_train)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                         batch_size=batch_size_new,
                                          drop_last=False, 
                                         shuffle=True)
                                         
    opt.n_batches_per_epoch = (n_train // batch_size_new)
    opt.lm = opt.lm / (np.sqrt(batch_grow_factor ** n_iters))
    print('LM regularization = ', opt.lm)

    return train_loader, opt