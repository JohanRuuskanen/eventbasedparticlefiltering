
Small package for state estimation using particle filters in systems with
event-based sampling.

Take a look in examples/ to get a feel on how to use the package.

=== Requirements === \n
This package was developed and tested for Julia 1.0.1

=== Add the package to Julia === \n
In your package manager in Julia (type "]" in your Julia console), type

> add path/to/package\n

=== To run an example ===
Running e.g. examples/filtering.jl is done by starting the Julia console typing

> using EventBasedParticleFiltering\n
> run_example("filtering.jl")\n

Don't run

> julia examples/filtering.jl

as the plotting windows will automatically close when the code finished.

=== To run the testing ===\n
Easiest done by activating the package and type test. This is done by opening
your package manager in Julia (type "]" in your Julia console) and write

> activate /path/to/packate\n
> test

=== Notice ===\n
This package is very much still a work in progress and thus there is still much
refactoring/testing and new features to implement.
