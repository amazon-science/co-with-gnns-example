## PI-GNN Example Code

In this notebook ([gnn_example.ipynb](gnn_example.ipynb)) we show how to solve combinatorial optimization problems 
with physics-inspired graph neural networks, as outlined in M. J. A. Schuetz, J. K. Brubaker, H. G. Katzgraber,
_Combinatorial Optimization with Physics-Inspired 
Graph Neural Networks, [arXiv:2107.01188](https://arxiv.org/abs/2107.01188). 
Here we focus on the canonical maximum independent set (MIS) problem, but our approach can easily be extended to 
other combinatorial optimization problems. For the actual implementation of the graph neural network we use the 
open-source ```dgl``` library. 

## Environment Setup

Please note we have provided a `requirements.txt` file, which defines the environment required to run this code. 
Because some of the packages are not available on default OSX conda channels, we have also provided suggested 
channels to find them on. These can be distilled into a single line as such:

> conda create -n \<environment_name\> python=3 --file requirements.txt -c conda-forge -c dglteam -c pytorch

## Code Execution

Once the virtual environment is established (see above), running the code is straightforward. From the parent folder, 
launch the notebook via 

> conda activate \<environment_name\>
> jupyter notebook gnn_example.ipynb

Once in the notebook, run the cells via 

`Cell` > `Run All` 

or 

`Kernel` > `Restart & Run All`

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the LICENSE file.

The sample code within this documentation is made available under the MIT-0 license. See the LICENSE-SAMPLECODE file.
