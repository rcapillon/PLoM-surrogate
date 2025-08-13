# PLoM-surrogate
Tools for constructing a surrogate model for a stochastic numerical model using Probabilistic Learning on Manifolds
in a small data context.

This package handles multi-dimensional data indexed by time (time series) or a pseudo-time (e.g. frequency) and 
parametrized by a number of deterministic or random control parameters.

It is appropriate to handle small datasets, learning the manifold structure on which the dataset is
concentrated and generating additional data points that lie on the learned manifold. It can then create a surrogate
model for a specific time-step (or all time-steps if one desires), computing new samples conditionally to a choice
of specific control parameters, computing the conditional mean, covariance or confidence interval.

## Installation
First, clone the repository in the folder of your choice using:
```
git clone git@github.com:rcapillon/PLoM-surrogate.git
```
Then, activate the virtual environment for your project and install this package and its requirements with:
```
cd PLoM-surrogate/
pip install .
```

## Running examples
**Note**: Example scripts are set to open 4 processes for multithreading and take a few minutes to complete, depending
on the machine used.

Open a terminal in the examples/ directory of the cloned repository, then you can run either of the two examples using:
```
python3 example_sinc.py
```
or
```
python3 example_cantilever_beam.py
```
This will produce plots saved in the directory from which the example scripts are ran.

### Example: Sinc function
This example uses a custom Sinc function (single output) with two control parameters and two random parameters. 
The original dataset contains 90 random time series.

|                                                   Original data                                                   |                                                Additional data                                                |                                                     Surrogate model                                                      |
|:-----------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/rcapillon/PLoM-surrogate/blob/main/readme_files/sinc_original_data.png" width="400"> | <img src="https://github.com/rcapillon/PLoM-surrogate/blob/main/readme_files/sinc_mcmc_data.png" width="400"> | <img src="https://github.com/rcapillon/PLoM-surrogate/blob/main/readme_files/sinc_surrogate_timeseries.png" width="400"> |

## Usage
First, you will need the following imports:
```
from PLoM_surrogate.data import Dataset
from PLoM_surrogate.generators import Generator
from PLoM_surrogate.models import Surrogate
```
Arrange your data in an array with shape (n_outputs + n_control, n_time, N), where 'n_outputs' is the number of 
components in the output vector at each time-step, 'n_control' is the number of control parameters, n_time is the number
of time-steps and N is the number of realizations (number of data points). Along the first dimension, the user must 
first put the outputs of the random model, then the values for the control parameters, copied for each time-step if
these are not time-dependent (note that you can have the control parameters be random variables themselves, just
organize the data correctly in the array).

Then, create a Dataset object with:
```
dataset = Dataset(data, n_outputs)
```
where 'data' is the data array described above.

The method requires to perform PCA on outputs and then again on the whole dataset, so, choose a number of principal 
components for the outputs, 'n_q', and do:
```
dataset.pca_on_Y(n_q)
dataset.full_pca_on_X()
```

Now, the dataset is ready in order to generate additional realizations, used to build a finer surrogate model than one
could get from the original small dataset. The algorithm will generate new realizations in chunks of N realizations, N
being the number of realizations in the original dataset. Thus, choose a 'n_MC' number of realizations chunks to 
generate, also choose a 'n_cpu' number of processes to use for multithreading. If you want to plot the eigenvalues used
to construct the reduced Diffusion Maps basis (related to the manifold learning), then add an additional string 
'plot_name' so the graph is saved on execution, otherwise don't include it. This can help choosing an appropriate number
of vectors to keep in the Diffusion Maps basis as recommended by [3] in the bibliography. 

If you have selected a number of vectors for the basis, add a 'm' argument at the creation of the generator:
```
generator = Generator(dataset, n_cpu, m=m)
generator.construct_dmaps_basis(plot_eigvals_name=plot_name)
```
This will generate the plot of the eigenvalues used to construct the reduced Diffusion Maps basis and can help you
choose the value for 'm'.

Then, generate the additional realizations using:
```
additional_data = generator.generate_realizations(n_MC)
```
The additional data will have the same shape as the data array arranged originally to create the dataset.

Finally, you can create a surrogate model with:
```
surrogate_model = Surrogate(additional_data, n_Y)
```
Create a numpy vector with the conditional values you want to use for the control parameters 'W_conditional'.
Also, choose a time-step index 'idx_t' picking the exact time-step you want to calculate quantities for then do:
```
surrogate_model.compute_surrogate_gkde(idx_t)
```
Finally, choose a number of samples 'n_s' used for estimations and a confidence level 'c_l' (e.g. c_l = 0.95) if you 
want to calculate the confidence interval.

You can then compute the conditional mean:
```
surrogate_mean = surrogate_model.compute_conditional_mean(W_conditional, n_s)
```
or the conditional covariance matrix:
```
surrogate_covar = surrogate_model.compute_conditional_covar(W_conditional, n_s)
```
or the confidence interval lower and upper bounds:
```
surrogate_lower_bound, surrogate_upper_bound = surrogate_model.compute_conditional_confidence_interval(
        W_conditional,
        n_s,
        c_l)
```
You can also simply generate additional samples conditioned on the value of the control parameters W:
```
new_samples = surrogate_model.conditional_sample(W_conditional, n_s)
```

## Bibliography
[1] C. Soize, R. Ghanem, 
Probabilistic-learning-based stochastic surrogate model from small incomplete datasets for nonlinear dynamical systems, 
Computer Methods in Applied Mechanics and Engineering, 2024, https://doi.org/10.1016/j.cma.2023.116498

[2] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations 
for small datasets, Computer Methods in Applied Mechanics and Engineering, 2021, 
https://doi.org/10.1016/j.cma.2021.113777

[3] C. Soize, R. Ghanem,
Data-driven probability concentration and sampling on manifold, Journal of Computational Physics, 2016,
https://doi.org/10.1016/j.jcp.2016.05.044

**Note**: this package does not implement incomplete observation of the output as in [1] and instead assumes that the
whole response of the model is known for every realization. Also, the original dataset is not constrained by an extra
dataset which may come from experimental data as in [2]. Reference [2] was used to implement the handling of time (or 
pseudo-time) dependent data.