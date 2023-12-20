// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#include "CMA_ES.h"

// PUBLIC METHODS
CMAES::CMAES(sim& s,
             constraints& c, 
             bopt& b, 
             int obj_fn, 
             const double* weights,
             std::string file_path) {

    // initialize input variables
    _pop = omp_get_num_procs();                                  // population size
    _P      = 4;                                                 // number of parents
    _C      = 4;                                                 // number of children
    _G      = 10;                                                // number of generations
    _obj_fn = obj_fn;                                            // objective function

    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    _param.resize(_pop, 10);

    // initialize simulation/optimization parameters and constraints
    _sim = s;
    _b   = b;
    _c   = c;

    // objective function weights
    _w   = new double[4];
    std::memcpy(_w, weights, 4 * sizeof(double));

    _file_path = file_path;
}

// Destructor
CMAES::~CMAES() {
    std::cout << "CMA_ES destructor called" << std::endl;
    delete[] _w;
}

void CMAES::initialize() {
    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // initials input samples
    std::cout << "--- INITIALIZING FIRST RUN ----" << std::endl;
    for (int i = 0; i < _param.rows(); ++i){
        _param(i, 0) = _c.min_temp + (_c.max_temp - _c.min_temp) * distribution(gen);
        _param(i, 1) = _c.min_rp   + (_c.max_rp   - _c.min_rp)   * distribution(gen);
        _param(i, 2) = _c.min_vp   + (_c.max_vp   - _c.min_vp)   * distribution(gen);
        _param(i, 3) = _c.min_uvi  + (_c.max_uvi  - _c.min_uvi)  * distribution(gen);
        _param(i, 4) = _c.min_uvt  + (_c.max_uvt  - _c.min_uvt)  * distribution(gen);
    }

    // initialize top performers
    #pragma omp parallel for
    for (int p = 0; p < _pop; ++p) {
        Voxel sim(_sim.tfinal,           // tot sim time
                  _sim.dt,               // time step
                  _sim.node,             // num nodes
                  _sim.time_stepping,    // sim id
                  _param(p, 0),          // amb temp
                  _param(p, 3),          // uv intensity
                  _param(p, 4),          // uv exposure time
                  _file_path,            // file path
                  true                   // multithread
                  );
        sim.computeParticles(_param(p, 1), _param(p, 2));
        sim.simulate(_sim.method,        // time stepping scheme
                    false,               // save voxel values
                    _obj_fn,             // objective function
                    _w                   // pareto weights
                  );
        #pragma omp critical
            {
                int thread_id = omp_get_thread_num();
                // std::cout << "Thread " << thread_id << std::endl;
                if (!std::isnan(sim.getObjective())) {
                    _param(p, 9) = sim.getObjective();
                    _param(p, 8) = sim.getObjM();
                    _param(p, 7) = sim.getObjMDot();
                    _param(p, 6) = sim.getObjPIDot();
                    _param(p, 5) = sim.getObjPI();
                } else {
                    _param(p, 9) = 1000.;
                    _param(p, 8) = 1000.;
                    _param(p, 7) = 1000.;
                    _param(p, 6) = 1000.;
                    _param(p, 5) = 1000.;

                }
            }
    }

    sort_data(_param);

    // track top and average performers
    _top_performer.push_back(_param(0, 9));
    _avg_parent.push_back(_param.col(9).head(_P).mean());
    _avg_total.push_back(_param.col(9).mean());
    _top_obj_pi.push_back(_param(0, 5));
    _top_obj_pidot.push_back(_param(0, 6));
    _top_obj_mdot.push_back(_param(0, 7));
    _top_obj_m.push_back(_param(0, 8));

    // track top decision variables
    _top_temp.push_back(_param(0, 0));
    _top_rp.push_back(_param(0, 1));
    _top_vp.push_back(_param(0, 2));
    _top_uvi.push_back(_param(0, 3));
    _top_uvt.push_back(_param(0, 4));

}

// PRIVATE METHODS
void CMAES::sort_data(Eigen::MatrixXd& param) {
    // Custom comparator for sorting by the fourth column in descending order
    auto comparator = [& param](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a(param.cols()-1) < b(param.cols()-1);
    };

    // Convert Eigen matrix to std::vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> rows;
    for (int i = 0; i < param.rows(); ++i) {
        rows.push_back(param.row(i));
    }

    // Sort using the custom comparator
    std::sort(rows.begin(), rows.end(), comparator);

    // Copy sorted rows back to Eigen matrix
    for (int i = 0; i < param.rows(); ++i) {
        param.row(i) = rows[i];
    }
}