// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#include "CMA_ES.h"

// PUBLIC METHODS
CMAES::CMAES(sim& s,
             constraints& c, 
             bopt& b, 
             int obj_fn,
             int n,
             const double* weights,
             std::string file_path) {

    _file_path = file_path;
    _obj_fn     = obj_fn;                                        // obj fcn
    _mthread    = true;                                          // multithread flag
    _save_voxel = false;                                         // save voxel values

    // initialize input variables
    _top_evals  = omp_get_num_procs();                           // num of evals per gen
    _m          = 8;                                             // cmaes pop size
    _m_elite    = 4;                                             // cmaes num parents
    _c          = _m_elite;                                      // ga num of children
    _G          = 10;                                            // num of gen
    _n          = n;                                             // num of vars

    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x (param + objs))
    _param_curr.resize(_m, 10);

    // initialize simulation/optimization parameters and constraints
    _sim   = s;
    _b     = b;
    _con   = c;

    // objective function weights
    _w   = new double[4];
    std::memcpy(_w, weights, 4 * sizeof(double));

    // allocate memory for statistical parameters
    _max_constraints.resize(_n);
    _min_constraints.resize(_n);

    // unpack constraints
    _max_constraints(0) = _con.max_temp;
    _max_constraints(1) = _con.max_rp;
    _max_constraints(2) = _con.max_vp;
    _max_constraints(3) = _con.max_uvi;
    _max_constraints(4) = _con.max_uvt;

    _min_constraints(0) = _con.min_temp;
    _min_constraints(1) = _con.min_rp;
    _min_constraints(2) = _con.min_vp;
    _min_constraints(3) = _con.min_uvi;
    _min_constraints(4) = _con.min_uvt;
}

// Destructor
CMAES::~CMAES() {
    std::cout << "CMA_ES destructor called" << std::endl;
    delete[] _w;
}

void CMAES::optimize() {
  
  // opt. parameters
  double sigma = .5;
  double sig_interm;

  // set sample weights
  Eigen::VectorXd ws(_m);
  for (int i = 0; i < _m; ++i) {
    ws[i] = std::log(_m + .5) - std::log(i + 1.);
  }
  float ws_sum = ws.sum();
  ws /= ws_sum;

  // initialize constants for adaptation
  double mu_eff = ws_sum * ws_sum / ws.squaredNorm();
  double cc     = (4. + mu_eff / _n) / (_n + 4. + 2. * mu_eff / _n);
  double cs     = (mu_eff + 2.) / (_n + mu_eff + 5.);
  double c1     = 2. / ((_n + 1.3) * (_n + 1.3) + mu_eff);
  double cmu    = std::min(1. - c1, 2. * (mu_eff - 2. + 1. / mu_eff) / ((_n + 2) * (_n + 2) + mu_eff));
  double ds     = 1. + 2. * std::max(0., std::sqrt((mu_eff - 1.) / (_n + 1.)) - 1.) + cs;
  int    hsig;

  // initialize dynamic (internal) strategy parameters and constants
  Eigen::VectorXd pc = Eigen::VectorXd::Zero(_n);                 // evolution path for C
  Eigen::VectorXd ps = Eigen::VectorXd::Zero(_n);                 // evolution path for C

  // initialize covariance matrix
  Eigen::MatrixXd B  = Eigen::MatrixXd::Identity(_n, _n);         // B defines the coordinate system
  Eigen::MatrixXd D  = Eigen::MatrixXd::Identity(_n, _n);         // diagonal matrix D defines the scaling
  Eigen::MatrixXd C  = B * D * (B * D).transpose();               // covariance matrix C

  double chiN        = std::sqrt(_n) * (1. - 1./(4.*_n) + 1./(21.*_n*_n));

  // generation loop
  int count_eval = 0;
  int eigen_eval = 0;
  Eigen::MatrixXd Xs(_n, _m);
  Eigen::MatrixXd Zs(_n, _m);

  Eigen::MatrixXd Xs_temp;
  Eigen::MatrixXd Zs_temp;
  Eigen::MatrixXd tmp;

  // initialize "data" as constraints
  Eigen::MatrixXd init_data(2, _n); 
  init_data << _con.max_temp, _con.max_rp, _con.max_vp, _con.max_uvi, _con.max_uvt,
                _con.min_temp, _con.min_rp, _con.min_vp, _con.min_uvi, _con.min_uvt;

  // compute mean
  Eigen::VectorXd xmu(_n), zmu(_n);
  xmu = init_data.colwise().mean();

  // begin optimization loop
  for (int g = 0; g < 5; ++g) {
    std::cout << "---- GENERATION " << g << " ----" << std::endl;

    // generate Zs and Xs
    this->gen_Zs(Zs);
    Xs = (sigma * B * D * Zs);
    Xs.colwise() += xmu;

    // clamp values at constraints
    Xs.row(0) = Xs.row(0).cwiseMax(_con.min_temp).cwiseMin(_con.max_temp);
    Xs.row(1) = Xs.row(1).cwiseMax(_con.min_rp).cwiseMin(_con.max_rp);
    Xs.row(2) = Xs.row(2).cwiseMax(_con.min_vp).cwiseMin(_con.max_vp);
    Xs.row(3) = Xs.row(3).cwiseMax(_con.min_uvi).cwiseMin(_con.max_uvi);
    Xs.row(4) = Xs.row(4).cwiseMax(_con.min_uvt).cwiseMin(_con.max_uvt);

    // copy Xs to param_curr
    _param_curr.block(0, 0, _m, _n) = Xs.transpose();
    
    // evaluate objective function
    #pragma omp parallel for 
    for (int m = 0; m < _m; m++) {
      // simulation call
      Voxel sim(_sim.tfinal,                // tot sim time
                _sim.dt,                    // time step
                _sim.node,                  // num nodes
                _sim.time_stepping,         // sim id
                _param_curr(m, 0),          // amb temp
                _param_curr(m, 3),          // uv intensity
                _param_curr(m, 4),          // uv exposure time
                _file_path,
                _mthread);

      // generate particles
      sim.computeParticles(_param_curr(m, 1),
                           _param_curr(m, 2));

      // run simulation
      sim.simulate(_sim.method,             // time stepping scheme
                   _save_voxel,             // save voxel values
                   _obj_fn,                 // objective function
                   _w                       // pareto weights
                  );
    
      // record objective values
      #pragma omp critical
      {
        int thread_id = omp_get_thread_num();
        if (!std::isnan(sim.getObjective())) {
          _param_curr(m, 9) = sim.getObjective();
          _param_curr(m, 8) = sim.getObjM();
          _param_curr(m, 7) = sim.getObjMDot();
          _param_curr(m, 6) = sim.getObjPIDot();
          _param_curr(m, 5) = sim.getObjPI();
        } else {
          _param_curr(m, 9) = 1.;
          _param_curr(m, 8) = 1.;
          _param_curr(m, 7) = 1.;
          _param_curr(m, 6) = 1.;
          _param_curr(m, 5) = 1.;
        }
      }
    }

    // sort by fitness
    this->sort_data(_param_curr);

    // copy ranked data back to Xs
    Xs = _param_curr.block(0, 0, _m, _n).transpose();

    xmu = Xs * ws;
    zmu = Zs * ws;

    // cumulation: update evolution paths
    ps   = (1. - cs) * ps + std::sqrt(cs * (2. - cs) * mu_eff) * B * zmu;
    hsig = ps.norm() / std::sqrt(1. - std::pow(1. - cs, 2. * count_eval / _m)) / chiN < 1.4 + 2. / (_n + 1.);
    pc   = (1. - cc) * pc + hsig * std::sqrt(cc * (2. - cc) * mu_eff) * (B * D * zmu);

    // adapt covariance matrix C
    tmp = B * D * Xs.block(0, 0, _n, _m_elite);
    C   =  (1. - c1 - cmu) * C
          + c1 * (pc * pc.transpose() + (1. - hsig) * cc * (2. - cc) * C)
          + cmu * tmp * ws.asDiagonal() * tmp.transpose();

    // check if sigma is too large
    sig_interm = sigma * std::exp((cs / ds) * (ps.norm() / chiN - 1.));
    if (sig_interm > 1e6) {
      throw std::invalid_argument("SIGMA TOO LARGE: terminating optimization");
    } else {
      sigma = sig_interm;
    }

    if (count_eval - eigen_eval > _m / (c1 + cmu) / _n * 0.1) {
      eigen_eval = count_eval;
      // enforce symmetry
      C = (C + C.transpose()) * 0.5;
      
      // eigen decomposition
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(C);
      D = eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
      B = eigen_solver.eigenvectors();
    }


    std::cout << "OBJECTIVES: " << _param_curr(0, 9) << std::endl;
    
    std::cout << "=========================\n" << std::endl;
    
    std::cout << std::endl;

  }
}

// PRIVATE METHODS
void CMAES::gen_Zs(Eigen::MatrixXd& Z) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> ndis(0.0, 1.0);
  std::uniform_real_distribution<> udis(0.0, 1.0);
  for (int i = 0; i < _m; ++i) {
      for (int j = 0; j < _n; ++j) {
        Z(j, i) = ndis(gen);
      }
    }
}

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