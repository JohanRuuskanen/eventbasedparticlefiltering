
# Script for starting remote workers on remote Linux machines, from a local
# Linux machine. Run before running the larger scripts to initialize a large
# enough work pool.
#
# Requires the same julia version on remote machines as for local machines. Also
# needs a python installation with matplotlib because of plotting via PyPlot

using Pkg
using Printf
using Distributed

# -------------------- SETUP OF MACHINES ---------------------------------
base_path = "/var/tmp/julia-tmpdir/"
depot_path = joinpath(base_path, ".julia")
project_path = joinpath(base_path, "project")
package_path = joinpath(base_path, "packages")
packages = ["Random", "Distributions", "LinearAlgebra", "JLD"]

remote_exepath = "julia"


# Change this to point at the manuscript_code directory
local_package = ("eventbasedparticlefiltering.jl", "/path/to/local/package")

# Change this, for each machine there should be a ("IP", number_of_workers) tuple
machines = [("IP-to-machine-1", 3),
            ("IP-to-machine-2", 3),
            ("etc...", 3)]

# Dirty-hack solution for spontanous errors that might occur when loading all
# those workers at the same time
tries = 5

remove_old_project_path = false

## First remove machines that can not be reached

for m in connection_error
    global machines
    filter!(x -> x[1] != m, machines)
end

if !isempty(connection_error)
    println("Failed to conenct to the follwing machines:")
    for m in connection_error
        println("\t $(m)")
    end
end

## Instatiate the remote workers

if nprocs() == 1

    # Create a new package path to store packages
    empty!(DEPOT_PATH)
    push!(DEPOT_PATH, depot_path)

    ENV["PYTHON"] = "python"

    # Copy local package to the correct path
    run(`rm -rf $(package_path)/$(local_package[1])`)
    run(`mkdir -p $(package_path)/$(local_package[1])`)
    run(`cp -r $(local_package[2]) $(package_path)/$(local_package[1])`)

    # Setup local project
    printstyled("Setting up local project:\n",bold=true,color=:magenta)
    Pkg.activate(project_path)
    Pkg.add(packages)
    Pkg.add(PackageSpec(path=joinpath(package_path, local_package[1])))

    # Transfer the necessary scripts
    for m in machines
        printstyled("Copying project to $(m[1]):\n",bold=true,color=:magenta)
        if remove_old_project_path
            run(`ssh -q -t $(m[1]) rm -rf $project_path`)
        end
        run(`ssh -q -t $(m[1]) mkdir -p $project_path`)
        run(`ssh -q -t $(m[1]) rm -rf $package_path`) # Overwrite
        run(`scp $project_path/Manifest.toml $(m[1]):$project_path`)
        run(`scp $project_path/Project.toml $(m[1]):$project_path`)
        println("Transferring local packages")
        run(`scp -q -r $package_path $(m[1]):$package_path`)
    end

    # hacky solution to precompilation issues
    machines_pcmp = [(m[1], 1) for m in machines]
    machines_rest = [(m[1], m[2]-1) for m in machines]

    # Setup workers
    printstyled("Starting precompilation workers on each designated machine:\n"
            ,bold=true,color=:magenta)

    addprocs(machines_pcmp,topology=:master_worker,tunnel=true,
        dir=project_path, exename=remote_exepath,
        max_parallel=24*length(machines))

    # Download, precompile and import required packages on remote machines
    printstyled("Downloading and compiling required packages to worker sessions:\n"
        ,bold=true,color=:magenta)

    @everywhere empty!(DEPOT_PATH)
    @everywhere push!(DEPOT_PATH, $depot_path)

    @everywhere ENV["PYTHON"] = "python"

    count = 0
    while count < tries
        try
            @everywhere begin
                using Pkg
                Pkg.activate($project_path)
                Pkg.instantiate()
                ps = vcat($packages, local_package[1])
                for i in ps
                    expr = :(using $(Symbol(i)))
                    eval(expr)
                end
            end
            break
        catch
            global count
            count += 1
            printstyled("Error occured, trying $(count) / $(tries)\n"
                ,bold=true,color=:red)
        end
    end

    printstyled("Starting rest of workers on each designated machine:\n"
            ,bold=true,color=:magenta)

    addprocs(machines_rest,topology=:master_worker,tunnel=true,
        dir=project_path, exename=remote_exepath,
        max_parallel=24*length(machines))

    # Download, precompile and import required packages on remote machines
    printstyled("Importing required packages to worker sessions:\n"
        ,bold=true,color=:magenta)

    @everywhere empty!(DEPOT_PATH)
    @everywhere push!(DEPOT_PATH, $depot_path)

    @everywhere ENV["PYTHON"] = "python"

    count = 0
    while count < tries
        try
            @everywhere begin
                using Pkg
                Pkg.activate($project_path)
                Pkg.instantiate()
                for i in $packages
                    expr = :(using $(Symbol(i)))
                    eval(expr)
                end
            end
            break
        catch
            global count
            count += 1
            printstyled("Error occured, trying $(count) / $(tries)\n"
                ,bold=true,color=:red)
        end
    end

    printstyled("Setup of workers on machines done!\n"
        ,bold=true,color=:magenta)

end
