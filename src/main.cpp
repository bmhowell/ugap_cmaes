// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#include "Voxel.h"
#include "CMA_ES.h"
#include "helper_functions.h"
#include "common.h"

void sort_data(Eigen::MatrixXd& PARAM);

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
    CMAES optimizer = CMAES(s,
                            c,
                            b,
                            obj_fn,
                            pareto_weights,
                            file_path);

    optimizer.initialize();

    optimizer.optimize();

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration = t.count() / 1e6;
    std::cout << "\n---Time taken by code segment: "
              << duration  / 60
              << " min---" << std::endl;
}

void sort_data(Eigen::MatrixXd& PARAM){
    // Custom comparator for sorting by the fourth column in descending order
    auto comparator = [& PARAM](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a(PARAM.cols()-1) < b(PARAM.cols()-1);
    };

    // Convert Eigen matrix to std::vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> rows;
    for (int i = 0; i < PARAM.rows(); ++i) {
        rows.push_back(PARAM.row(i));
    }

    // Sort using the custom comparator
    std::sort(rows.begin(), rows.end(), comparator);

    // Copy sorted rows back to Eigen matrix
    for (int i = 0; i < PARAM.rows(); ++i) {
        PARAM.row(i) = rows[i];
    }
}
