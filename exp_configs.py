import os
import itertools

from haven import haven_utils as hu

sgd_armijo_list =  [{"name":"sgd_armijo", "eta_max":1, "gamma":1.5}]
others_list = [{'name':"adam"}, {'name':"svrg"}, {'name':"adagrad"}]
sgd_polyak_list = [{"name":"sgd_polyak", "c": 0.1, "momentum": 0.6, "eta_bound":1, "gamma":1.5, "reset":1}]

opt_list = others_list + sgd_polyak_list + sgd_armijo_list

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


EXP_GROUPS = {
        # synthetic experiments
        "syn_basic":hu.cartesian_exp_group({"dataset":syn_datasets,
                "model":model,
                "loss_func": loss_list,
                "acc_func": ["logistic_accuracy"],
                "n_samples": syn_n_samples,
                "d": syn_dims,
                "opt": opt_list + [{"name":"ssn", "lm":1e-3},
                                   {"name":"slbfgs", 'line_search_fn':'sls', 'lr':0.9, "lm":1e-4, "history_size":10}],
                "margin":margin_list,
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list}),


        "syn_grow":hu.cartesian_exp_group({"dataset":syn_datasets,
                "model":model,
                "loss_func": loss_list,
                "acc_func": ["logistic_accuracy"],
                "n_samples": syn_n_samples,
                "d": syn_dims,
                "opt":[{"name":"ssn","lm":1e-6}],
                "margin":margin_list,
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list,
                "batch_grow_factor":batch_grow_factor,
                "batch_size_max":batch_size_max,}),


        "syn_full":hu.cartesian_exp_group({"dataset":syn_datasets,
                "model":model,
                "loss_func": loss_list,
                "acc_func": ["logistic_accuracy"],
                "n_samples": syn_n_samples,
                "d": syn_dims,
                "opt":[{"name":"ssn","lm":0}, 
                       {"name":"lbfgs", "history_size":10, "max_iter":2}],
                "margin":margin_list,
                "batch_size":["full"],
                "max_epoch":max_epoch,
                "runs":run_list}),      

        # kernel experiments
        "kernels_basic":hu.cartesian_exp_group({"dataset":kernel_datasets,
                "model":model,
                "loss_func": loss_list,
                "acc_func": ["logistic_accuracy"],
                "opt":opt_list + [{"name":"ssn", "lm":1e-3}, {"name":"slbfgs", "lm":0, "history_size":10, "lr":0.1}],
                "batch_size":[100],
                "max_epoch":max_epoch,
                "runs":run_list}),


        "kernels_grow":hu.cartesian_exp_group({"dataset":kernel_datasets,
                "model":model,
                "loss_func": loss_list,
                "acc_func": ["logistic_accuracy"],
                # "opt":[{"name":"ssn", "lm":lm} for lm in [0,1e-3, 1e-4,1e-6]],
                "opt":[{"name":"ssn", "lm":1e-3}],
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