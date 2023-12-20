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


    bopt default_bopt;
    default_bopt.temp = 303.15;
    default_bopt.rp   = 0.00084 / 10;
    default_bopt.vp   = 0.7;
    default_bopt.uvi  = 10.;
    default_bopt.uvt  = 30.;
    
    sim default_sim;
    default_sim.time_stepping = 0;
    default_sim.update_time_stepping_values();
    const bool mthread = true; 
    int   save_voxel = 0;
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_cmaes/output_" 
                            + std::to_string(default_sim.time_stepping);
    
    // std::string file_path = "/home/brian/Documents/brian/ugap_cmaes/output_"
	//                     + std::to_string(default_sim.time_stepping); 
    
    // // GA parameters
    // int pop = omp_get_num_procs();                                  // population size
    // int P   = 4;                                                    // number of parents
    // int C   = 4;                                                    // number of children
    // int G   = 10;                                                   // number of generations
    // double lam_1, lam_2;                                            // genetic alg paramters

    // // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    // Eigen::MatrixXd param(pop, 10);        

    // // initialize input variables
    // std::random_device rd;                                          // Obtain a random seed from the hardware
    // std::mt19937 gen(rd());                                         // Seed the random number generator
    // std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // // initials input samples
    // std::cout << "--- INITIALIZING FIRST RUN ----" << std::endl;
    // for (int i = 0; i < param.rows(); ++i){
    //     param(i, 0) = c.min_temp + (c.max_temp - c.min_temp) * distribution(gen);
    //     param(i, 1) = c.min_rp   + (c.max_rp   - c.min_rp)   * distribution(gen);
    //     param(i, 2) = c.min_vp   + (c.max_vp   - c.min_vp)   * distribution(gen);
    //     param(i, 3) = c.min_uvi  + (c.max_uvi  - c.min_uvi)  * distribution(gen);
    //     param(i, 4) = c.min_uvt  + (c.max_uvt  - c.min_uvt)  * distribution(gen);
    // }

    // // performance vectors
    // std::vector<double> top_performer; 
    // std::vector<double> avg_parent; 
    // std::vector<double> avg_total;
    // std::vector<double> top_obj_pi, top_obj_pidot, top_obj_mdot, top_obj_m;
    // std::vector<double> top_temp, top_rp, top_vp, top_uvi, top_uvt;

    // // initialize top performers
    // #pragma omp parallel for
    // for (int p = 0; p < pop; ++p) {
    //     Voxel sim(default_sim.tfinal,    // tot sim time
    //               default_sim.dt,        // time step
    //               default_sim.node,      // num nodes
    //               default_sim.time_stepping,    // sim id
    //               param(p, 0),           // amb temp
    //               param(p, 3),           // uv intensity
    //               param(p, 4),           // uv exposure time
    //               file_path,
    //               mthread);
    //     sim.computeParticles(param(p, 1), param(p, 2));
    //     sim.simulate(default_sim.method,    // time stepping scheme
    //                 save_voxel,             // save voxel values
    //                 obj_fn,                 // objective function
    //                 pareto_weights          // pareto weights
    //               );
    //     #pragma omp critical
    //         {
    //             int thread_id = omp_get_thread_num();
    //             // std::cout << "Thread " << thread_id << std::endl;
    //             if (!std::isnan(sim.getObjective())) {
    //                 param(p, 9) = sim.getObjective();
    //                 param(p, 8) = sim.getObjM();
    //                 param(p, 7) = sim.getObjMDot();
    //                 param(p, 6) = sim.getObjPIDot();
    //                 param(p, 5) = sim.getObjPI();
    //             } else {
    //                 param(p, 9) = 1000.;
    //                 param(p, 8) = 1000.;
    //                 param(p, 7) = 1000.;
    //                 param(p, 6) = 1000.;
    //                 param(p, 5) = 1000.;

    //             }
    //         }
    // }

    // sort_data(param);

    // // track top and average performers
    // top_performer.push_back(param(0, 9));
    // avg_parent.push_back(param.col(9).head(P).mean());
    // avg_total.push_back(param.col(9).mean());
    // top_obj_pi.push_back(param(0, 5));
    // top_obj_pidot.push_back(param(0, 6));
    // top_obj_mdot.push_back(param(0, 7));
    // top_obj_m.push_back(param(0, 8));

    // // track top decision variables
    // top_temp.push_back(param(0, 0));
    // top_rp.push_back(param(0, 1));
    // top_vp.push_back(param(0, 2));
    // top_uvi.push_back(param(0, 3));
    // top_uvt.push_back(param(0, 4));


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
