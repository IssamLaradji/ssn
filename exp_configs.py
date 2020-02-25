import os
import itertools

from haven import haven_utils as hu

# sgd_armijo_list =  [{"name":"sgd_armijo", "eta_max":1, "gamma":1.5}]
others_list = [{'name':"adam"}, {'name':"svrg"}, {'name':"adagrad"}]
# sgd_polyak_list = [{"name":"sgd_polyak", "c": 0.1, "momentum": 0.6, "eta_bound":1, "gamma":1.5, "reset":1}]

opt_list = others_list

syn_datasets = ["synthetic"]
syn_n_samples = [10000]
syn_dims = [20]
margin_list = [0.01, 0.05, 0.1, 0.5]
kernel_datasets = ["mushrooms", "ijcnn", "rcv1"]

loss_list = [ 
       "squared_hinge_loss", 
       "logistic_loss"
       ]

model = ["linear"]
batch_grow_factor = [1.01]
batch_size_max = [8192]
max_epoch = [200]

run_list = [0,1,2,3,4]

def get_kernel_exp_list(dataset, loss, 
                        ssn_const_lm,
                        ssn_grow_lm,  
                        slbfgs_lr, 
                        slbfgs_lm):
    exp_list = (
        
    hu.cartesian_exp_group({
        "dataset":dataset,
        "model":model,
        "loss_func": loss,
        "acc_func": ["logistic_accuracy"],
        "opt":[{"name":"ssn", "lm":ssn_grow_lm}],
        "batch_size":[100],
        "max_epoch":max_epoch,
        "runs":run_list,
        "batch_grow_factor":batch_grow_factor,
        "batch_size_max":batch_size_max,})
        +
        hu.cartesian_exp_group(
    {"dataset":dataset,
        "model":model,
        "loss_func": loss,
        "acc_func": ["logistic_accuracy"],
        "opt":opt_list + [
            {"name":"ssn", "lm":ssn_const_lm}, 
        {"name":"slbfgs", "lm":slbfgs_lm, "history_size":10, "lr":slbfgs_lr}],
        "batch_size":[100],
        "max_epoch":max_epoch,
        "runs":run_list})
            )

    return exp_list

def get_ijcnn_grid_search():
    exp_list = hu.cartesian_exp_group({
        "dataset":'ijcnn',
        "model":model,
        "loss_func": ['logistic_loss', 'squared_hinge_loss'],
        "acc_func": ["logistic_accuracy"],
        "opt":[{"name":"ssn", "lm":lm} for lm in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)],
        "batch_size":[100],
        "max_epoch":max_epoch,
        "runs":run_list,
        "batch_grow_factor":batch_grow_factor,
        "batch_size_max":batch_size_max,})
        

    return exp_list

def get_syn_exp_list(loss):
    exp_list = hu.cartesian_exp_group({"dataset":syn_datasets,
                "model":model,
                "loss_func": loss,
                "acc_func": ["logistic_accuracy"],
                "n_samples": syn_n_samples,
                "d": syn_dims,
                "opt": opt_list + [{"name":"ssn", "lm":1e-3},
                                   {"name":"slbfgs", 'line_search_fn':'sls', 'lr':0.9, "lm":1e-4, "history_size":10}],
                "margin":margin_list,
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list})
    exp_list += hu.cartesian_exp_group({"dataset":syn_datasets,
                "model":model,
                "loss_func": loss,
                "acc_func": ["logistic_accuracy"],
                "n_samples": syn_n_samples,
                "d": syn_dims,
                "opt":[{"name":"ssn","lm":1e-6}],
                "margin":margin_list,
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list,
                "batch_grow_factor":batch_grow_factor,
                "batch_size_max":batch_size_max,})
    exp_list += hu.cartesian_exp_group({"dataset":syn_datasets,
                "model":model,
                "loss_func": loss,
                "acc_func": ["logistic_accuracy"],
                "n_samples": syn_n_samples,
                "d": syn_dims,
                "opt":[{"name":"ssn","lm":0}, 
                       {"name":"lbfgs", "history_size":10, "max_iter":2}],
                "margin":margin_list,
                "batch_size":["full"],
                "max_epoch":max_epoch,
                "runs":run_list})
    return exp_list

EXP_GROUPS = {
        # synthetic experiments
        "syn_logistic":get_syn_exp_list('logistic_loss'),
        "syn_squared_hinge":get_syn_exp_list('squared_hinge_loss'),


        # kernel experiments
        "ssn_const":hu.cartesian_exp_group({"dataset":['ijcnn'],
                "model":model,
                "loss_func": ['logistic_loss'],
                "acc_func": ["logistic_accuracy"],
                "opt": [{"name":"ssn", "lm":1e-3}],
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list}),
        "mushrooms_logistic":get_kernel_exp_list(dataset='mushrooms', 
                                                 loss='logistic_loss', 
                                                 ssn_const_lm=1e-3,  
                                                 ssn_grow_lm=1e-4, 
                                                 slbfgs_lr=0.5, 
                                                 slbfgs_lm=1e-4
                                                 ),
        "mushrooms_squared_hinge":get_kernel_exp_list(dataset='mushrooms', 
                                                 loss='squared_hinge_loss', 
                                                 ssn_const_lm=1e-3,  
                                                 ssn_grow_lm=1e-3, 
                                                 slbfgs_lr=0.5, 
                                                 slbfgs_lm=0
                                                 ),
        "ijcnn_grid_search":get_ijcnn_grid_search(),
         "ijcnn_logistic":get_kernel_exp_list(dataset='ijcnn', 
                                                 loss='logistic_loss', 
                                                 ssn_const_lm=1e-3,  
                                                 ssn_grow_lm=1e-4, 
                                                 slbfgs_lr=0.5, 
                                                 slbfgs_lm=1e-4
                                                 ),
        "ijcnn_squared_hinge":get_kernel_exp_list(dataset='ijcnn', 
                                                 loss='squared_hinge_loss', 
                                                 ssn_const_lm=1e-3,  
                                                 ssn_grow_lm=1e-3, 
                                                 slbfgs_lr=0.5, 
                                                 slbfgs_lm=0
                                                 ),

        "rcv1_logistic":get_kernel_exp_list(dataset='rcv1', 
                                                 loss='logistic_loss', 
                                                 ssn_const_lm=1e-3,  
                                                 ssn_grow_lm=1e-3, 
                                                 slbfgs_lr=0.1, 
                                                 slbfgs_lm=1e-4
                                                 ),
        "rcv1_squared_hinge":get_kernel_exp_list(dataset='rcv1', 
                                                 loss='squared_hinge_loss', 
                                                 ssn_const_lm=1e-3,  
                                                 ssn_grow_lm=1e-3, 
                                                 slbfgs_lr=0.1, 
                                                 slbfgs_lm=0
                                                 ),
        # "rcv1_logistic":
        # "rcv1_squared_hinge":

        "kernels_grow":hu.cartesian_exp_group({"dataset":kernel_datasets,
                "model":model,
                "loss_func": loss_list,
                "acc_func": ["logistic_accuracy"],
                # "opt":[{"name":"ssn", "lm":lm} for lm in [0, 1e-3, 1e-4, 1e-6]],
                "opt":[
                    {"name":"ssn", "lm":1e-3}, {"name":"ssn", "lm":1e-4}],
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list,
                "batch_grow_factor":batch_grow_factor,
                "batch_size_max":batch_size_max,}),

            }

# grid search results
step_sizes = {
    "logistic_loss" : {
        "svrg" : {
            'mushrooms' : 500., 
            'rcv1'      : 500.,
            "ijcnn"     : 500.,
            "w8a"       : 0.0025,
            'syn-0.01'  : 1.5,
            'syn-0.05'  : 0.1,
            'syn-0.1'   : 0.025,
            'syn-0.5'   : 0.0025,
            'syn-1.0'   : 0.001,
        },
       

    },
    "squared_hinge_loss" : {
        "svrg" : {
            'mushrooms' : 150., 
            'rcv1'      : 3.25,
            "ijcnn"     : 2.75,
            "w8a"       : 0.00001,
            'syn-0.01'  : 1.25,
            'syn-0.05'  : 0.025,
            'syn-0.1'   : 0.0025,
            'syn-0.5'   : 0.001,
            'syn-1.0'   : 0.001,
        },
    }
}