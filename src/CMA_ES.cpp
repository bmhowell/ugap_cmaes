// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#include "CMA_ES.h"

// PUBLIC METHODS
CMAES::CMAES(sim& s,
             constraints& c, 
             bopt& b, 
             int obj_fn,
             int n_var,
             const double* weights,
             std::string file_path) {

    _file_path = file_path;

    // initialize input variables
    _pop        = omp_get_num_procs();                           // population size
    _P          = 4;                                             // number of parents
    _C          = 4;                                             // number of children
    _G          = 10;                                            // number of generations
    _obj_fn     = obj_fn;                                        // objective function
    _n_var      = n_var;                                         // number of variables
    _mthread    = true;                                          // multithread flag
    _save_voxel = false;                                         // save voxel values

    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    _param_curr.resize(_pop, 10);
    _param_next.resize(_pop, 10);

    // initialize simulation/optimization parameters and constraints
    _sim = s;
    _b   = b;
    _c   = c;

    // objective function weights
    _w   = new double[4];
    std::memcpy(_w, weights, 4 * sizeof(double));

    // allocate memory for statistical parameters
    normed_data = false;
    _min_param.resize(_n_var);
    _max_param.resize(_n_var);
    _mu_curr.resize(_n_var);
    _mu_next.resize(_n_var);
    _sigma.resize(_n_var, _n_var);

}

// Destructor
CMAES::~CMAES() {
    std::cout << "CMA_ES destructor called" << std::endl;
    delete[] _w;
}

void CMAES::initialize() {
  // set random device and generator
  std::random_device rd;                                          // Obtain a random seed from the hardware
  std::mt19937 gen(rd());                                         // Seed the random number generator
  std::uniform_real_distribution<double> udis(0.0, 1.0);          // Define the range [0.0, 1.0)

  // generate random samples for bootstrap
  std::cout << "---- INITIALIZING FIRST RUN ----" << std::endl;
  for (int i = 0; i < _pop; ++i){
    _param_curr(i, 0) = _c.min_temp + (_c.max_temp - _c.min_temp) * udis(gen);
    _param_curr(i, 1) = _c.min_rp   + (_c.max_rp   - _c.min_rp)   * udis(gen);
    _param_curr(i, 2) = _c.min_vp   + (_c.max_vp   - _c.min_vp)   * udis(gen);
    _param_curr(i, 3) = _c.min_uvi  + (_c.max_uvi  - _c.min_uvi)  * udis(gen);
    _param_curr(i, 4) = _c.min_uvt  + (_c.max_uvt  - _c.min_uvt)  * udis(gen);
  }

  // initialize top performers
  #pragma omp parallel for
  for (int p = 0; p < _pop; ++p) {
    Voxel sim(_sim.tfinal,                // tot sim time
              _sim.dt,                    // time step
              _sim.node,                  // num nodes
              _sim.time_stepping,         // sim id
              _param_curr(p, 0),          // amb temp
              _param_curr(p, 3),          // uv intensity
              _param_curr(p, 4),          // uv exposure time
              _file_path,
              _mthread);
    sim.computeParticles(_param_curr(p, 1), _param_curr(p, 2));
    sim.simulate(_sim.method,             // time stepping scheme
                 _save_voxel,             // save voxel values
                 _obj_fn,                 // objective function
                 _w                       // pareto weights
                );

    #pragma omp critical
    {
      int thread_id = omp_get_thread_num();
      // std::cout << "Thread " << thread_id << std::endl;
      if (!std::isnan(sim.getObjective())) {
        _param_curr(p, 9) = sim.getObjective();
        _param_curr(p, 8) = sim.getObjM();
        _param_curr(p, 7) = sim.getObjMDot();
        _param_curr(p, 6) = sim.getObjPIDot();
        _param_curr(p, 5) = sim.getObjPI();
      } else {
        _param_curr(p, 9) = 1000.;
        _param_curr(p, 8) = 1000.;
        _param_curr(p, 7) = 1000.;
        _param_curr(p, 6) = 1000.;
        _param_curr(p, 5) = 1000.;
      }
    }
  }

  std::cout << "_param_curr: \n" << _param_curr << std::endl;

  // rank candidates
  sort_data(_param_curr);

  // store top performers
  _top_performer.push_back(_param_curr(0, 9));
  _top_obj_m.push_back(_param_curr(0, 8));
  _top_obj_mdot.push_back(_param_curr(0, 7));
  _top_obj_pidot.push_back(_param_curr(0, 6));
  _top_obj_pi.push_back(_param_curr(0, 5));

  // store top performer parameters
  _top_temp.push_back(_param_curr(0, 0));
  _top_rp.push_back(_param_curr(0, 1));
  _top_vp.push_back(_param_curr(0, 2));
  _top_uvi.push_back(_param_curr(0, 3));
  _top_uvt.push_back(_param_curr(0, 4));

  // store average top _P performers
  _avg_parent.push_back(_param_curr.col(9).head(_P).mean());

  // store average total population
  _avg_total.push_back(_param_curr.col(9).mean());

  std::cout << "_param_curr: \n" << _param_curr << std::endl;
}

void CMAES::optimize() {
    
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

void CMAES::norm_data(Eigen::MatrixXd& param) {
    // check if data is already normalized
    if (normed_data) { return; }

    // normalize data
    for (int i = 0; i < param.cols(); ++i) {
        // store min/max value for each variable
        _min_param(i) = param.col(i).minCoeff();
        _max_param(i) = param.col(i).maxCoeff();
        
        param.col(i) = (param.col(i).array() - _min_param(i)) / (_max_param(i) - _min_param(i));
    }

    // set flag
    normed_data = true;
}

void CMAES::unnorm_data(Eigen::MatrixXd& param) {
    // check if data is already unnormalized
    if (!normed_data) { return; }

    // unnormalize data
    for (int i = 0; i < param.cols(); ++i) {
        param.col(i) = param.col(i).array() * (_max_param(i) - _min_param(i)) + _min_param(i);
    }

    // set flag
    normed_data = false;
}
