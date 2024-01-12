// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#include "Voxel.h"
#include "CMA_ES.h"
#include "helper_functions.h"
#include "common.h"

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    // opt constrazints (default) and sim settings (default)
    constraints c;

    // CHOOSE OBJECTIVE FUNCTION
    // 1: PI, 2: PIdot, 3: Mdot, 4: M, 5: total
    int obj_fn = 5;
    double default_weights[4] = {0.1, 0.2, 0.2, 0.5};
    double pareto_weights[4]  = {3.56574286e-09, 2.42560512e-03, 2.80839829e-01, 7.14916061e-01};


    bopt b;
    b.temp = 303.15;
    b.rp   = 0.00084 / 10;
    b.vp   = 0.7;
    b.uvi  = 10.;
    b.uvt  = 30.;
    
    sim s;
    s.time_stepping = 0;
    s.update_time_stepping_values();
    int   save_voxel = 0;
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_cmaes/output_" 
                            + std::to_string(s.time_stepping);
    
    // std::string file_path = "/home/brian/Documents/brian/ugap_cmaes/output_"
	//                     + std::to_string(default_sim.time_stepping); 
    
    // initialize CMA-ES
    CMAES optimizer = CMAES(s,                  // simulation parameters
                            c,                  // optimization constraints
                            b,                  // optimization parameters
                            obj_fn,             // objective function
                            5,                  // number of variables
                            pareto_weights,     // objective function weights
                            file_path           // file path for output
                            );

    // optimizer.initialize();

    optimizer.initialize();
    // optimizer.optimize();

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration = t.count() / 1e6;
    std::cout << "\n---Time taken by code segment: "
              << duration  / 60
              << " min---" << std::endl;
}

