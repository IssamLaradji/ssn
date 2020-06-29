

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">SSN </h1>
<h5 align="center">Stochastic Second Order Methods under Interpolation (AISTATS 2020).</h5>


SSN [[Paper]](https://arxiv.org/pdf/1910.04920.pdf). 


### 1. Install requirements
Install the ssn optimizer.

`pip install git+https://github.com/IssamLaradji/ssn.git`


Install the [Haven library](https://github.com/ElementAI/haven) for managing the experiments.

`pip install -r requirements.txt`

### 2. Usage
Use `Ssn` in your code by adding the following script.

```python
import ssn
opt = ssn.Ssn(model.parameters())

for epoch in range(100):
    opt.zero_grad()
    closure = lambda : torch.nn.MSELoss() (model(X), Y)
    opt.step(closure=closure)
```

### 3. Experiments

Run an a synthetic experiment with the logistic loss with the command below,

`python trainval.py -e syn_logistic -sb ../results -r 1`

where `-e` is the experiment group, `-sb` is the result directory.

Other experiment groups are defined in `exp_configs.py`, which are the following:

- "syn_squared_hinge" 
- "mushrooms_logistic"
- "mushrooms_squared_hinge"
- "ijcnn_logistic"
- "ijcnn_squared_hinge"
- "rcv1_logistic"
- "rcv1_squared_hinge"

#### Citation

```
@inproceedings{meng2020fast,
  title={Fast and furious convergence: Stochastic second order methods under interpolation},
  author={Meng, Si Yi and Vaswani, Sharan and Laradji, Issam Hadj and Schmidt, Mark and Lacoste-Julien, Simon},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1375--1386},
  year={2020}
}
```