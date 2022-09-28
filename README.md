## Code for arXiv preprint

this document should get you started with generating the results shown in the preprint

> Johan Ruuskanen, Anton Cervin, "Improved event-based particle filtering in resource-constrained remote state estimation", 2022

#### How to run

Needed software

* `julia`, the code was developed using version 1.5.0, and it is recommended that the same version is used to assure that everything works as it should.
* `python` with `matplotlib.pyplot` installed, as the code uses PyPlot to display the results. 

Load the package for usage, start `julia` in the `manuscript_code` folder and run

```julia
using Pkg

ENV['PYTHON'] = 'python'

Pkg.activate(".")
Pkg.instantiate()

# To run the tests
Pkg.test()
```

Check out the scripts in `scripts/` for small scripts demonstrating the package. 

In `scripts_large/` the larger simulations can be found. These are not recommended to run on a single computer, but on a cluster. The script `scripts_large/start_remote.jl` can help to get you started.

For the larger simulations that generate the results displayed in the manuscript, the data has been provided from our runs. It can be loaded and displayed by running the bottom halves of the supplied scripts.
