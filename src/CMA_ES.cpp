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
    _m          = omp_get_num_procs();                           // population size
    _m_elite    = 4;                                             // number of parents
    _c          = 4;                                             // number of children
    _G          = 10;                                            // number of generations
    _sigma      = 1.0;                                           // step size
    _obj_fn     = obj_fn;                                        // objective function
    _n      = n_var;                                         // number of variables
    _mthread    = true;                                          // multithread flag
    _save_voxel = false;                                         // save voxel values

    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    _param_curr.resize(_m, 10);
    _param_next.resize(_m, 10);
    _Xs.resize(_n, _m);

    // initialize simulation/optimization parameters and constraints
    _sim   = s;
    _b     = b;
    _con   = c;

    // objective function weights
    _w   = new double[4];
    std::memcpy(_w, weights, 4 * sizeof(double));

    // allocate memory for statistical parameters
    _stdzd = false;
    _stdz_avg.resize(_n);
    _stdz_std.resize(_n);
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

void CMAES::initialize() {
  // set up random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> ndis(0.0,  1.0);
  std::uniform_real_distribution<double> udis(0.0, 1.0);

  // generate random samples for bootstrap
  std::cout << "---- INITIALIZING FIRST RUN ----" << std::endl;
  for (int i = 0; i < _m; ++i){
    _param_curr(i, 0) = _con.min_temp + (_con.max_temp - _con.min_temp) * udis(gen);
    _param_curr(i, 1) = _con.min_rp   + (_con.max_rp   - _con.min_rp)   * udis(gen);
    _param_curr(i, 2) = _con.min_vp   + (_con.max_vp   - _con.min_vp)   * udis(gen);
    _param_curr(i, 3) = _con.min_uvi  + (_con.max_uvi  - _con.min_uvi)  * udis(gen);
    _param_curr(i, 4) = _con.min_uvt  + (_con.max_uvt  - _con.min_uvt)  * udis(gen);
  }

  // initialize top performers
  #pragma omp parallel for
  for (int p = 0; p < _m; ++p) {
    Voxel sim(_sim.tfinal,                // tot sim time
              _sim.dt,                    // time step
              _sim.node,                  // num nodes
              _sim.time_stepping,         // sim id
              _param_curr(p, 0),          // amb temp
              _param_curr(p, 3),          // uv intensity
              _param_curr(p, 4),          // uv exposure time
              _file_path,
              _mthread);
    sim.computeParticles(_param_curr(p, 1),
                         _param_curr(p, 2));
    sim.simulate(_sim.method,             // time stepping scheme
                 _save_voxel,             // save voxel values
                 _obj_fn,                 // objective function
                 _w                       // pareto weights
                );

    #pragma omp critical
    {
      int thread_id = omp_get_thread_num();
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

  // rank candidates
  this->sort_data(_param_curr);

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

  // store average top _m_elite performers
  _avg_parent.push_back(_param_curr.col(9).head(_m_elite).mean());

  // store average total population
  _avg_total.push_back(_param_curr.col(9).mean());

  // standardize data
  this->stdz_data(_param_curr);

  // initialize covariance matrix
  Eigen::MatrixXd x_centered(_n, _m);
  _Xs  = _param_curr.block(0, 0, _m, _n).transpose();      // stdzd, transposed data ∈ ℝ(n_var x m)
  _mu  = _Xs.rowwise().mean();                                 // mean vector ∈ ℝ(n_var)
  x_centered = _Xs.colwise() - _mu;                            // centered data ∈ ℝ(n_var x m)
  _Cov = (x_centered * x_centered.transpose()) / (_m - 1);     // covariance matrix ∈ ℝ(n_var x n_var)
  std::cout << "Covariance matrix: \n" << _Cov << std::endl;
}

void CMAES::optimize() {
  

  // set sample wieghts
  Eigen::VectorXd ws(_m);
  for (int i = 0; i < _m; ++i) {
    ws(i) = std::log((_m + 1) / 2) - std::log(i + 1);
  }
  float ws_sum = ws.head(_m_elite).sum();
  ws.head(_m_elite) /= ws_sum;

  double mu_eff = 1.0 / ws.head(_m_elite).squaredNorm();
  double cs     = (mu_eff + 2.0) / (_n + mu_eff + 5.0);
  double ds     = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (_n + 1.0)) - 1.0) + cs;
  double cc     = (4. + mu_eff/_n) / (_n + 4. + 2. * mu_eff/_n);
  double c1     = 2. / ((_n+1.3)*(_n+1.3) + mu_eff);
  double cmu    = std::min(1. - c1, 2. * (mu_eff - 2. + 1./mu_eff) / ((_n+2)*(_n+2) + mu_eff));
  double E      = std::sqrt(_n) * (1. - 1./(4.*_n) + 1./(21.*_n*_n));
  int    hsig;
  double inv_sigma;

  ws.tail(_m - _m_elite) *= -(1. + c1/cmu) / ws.tail(_m - _m_elite).sum();
  
  // initialize mean
  Eigen::VectorXd mu(_n);
  for (int i = 0; i < _n; ++i) {
    mu(i) = (_max_constraints(i) + _min_constraints(i)) * 0.5;
  }
  _stdz_avg = mu;

  // initialize stddev
  for (int i = 0; i < _n; ++i) {
    _stdz_std(i) = std::sqrt(  ((_max_constraints(i) - mu(i)) * (_max_constraints(i) - mu(i))
                             +  (_min_constraints(i) - mu(i)) * (_min_constraints(i) - mu(i)))
                              * 0.5
                            );
  }

  // scale 

  Eigen::VectorXd ps = Eigen::VectorXd::Zero(_n);
  Eigen::VectorXd pc = Eigen::VectorXd::Zero(_n);
  Eigen::MatrixXd C(_n, _n);
  Eigen::MatrixXd Xs(_n, _m);
  Eigen::MatrixXd Zs(_n, _m);
  Eigen::MatrixXd delta_xs = Eigen::MatrixXd::Zero(_n, _m);
  Eigen::VectorXd delta_ws(_n);
  Eigen::VectorXd w0(_m);

  // initialize cov matrix
  this->init_Cov(C, mu);
  
  std::cout << "C: \n" << C << std::endl;

  // begin optimization loop
  for (int g = 0; g < 1; ++g) {
    std::cout << "---- GENERATION " << g << " ----" << std::endl;
    // cholseky decomposition
    Eigen::LLT<Eigen::MatrixXd> lltOfC(C);
    if (lltOfC.info() == Eigen::NumericalIssue) { std::cerr << "LLT failed!" << std::endl; }
    Eigen::MatrixXd L = lltOfC.matrixL();

    std::cout << "L: \n" << L << std::endl;

    // sample from multivariate normal distribution
    std::cout << "mu: " << mu.transpose() << std::endl;
    
    // generate Zs
    this->gen_Zs(Zs);
    

    std::cout << "Zs: " << Zs << std::endl;
    std::cout << std::endl << std::endl;
    Xs = (_sigma * L * Zs).colwise() + mu;
    std::cout << "before _Xs: \n" << Xs << std::endl;

    // cap values at constraints
    std::cout << "min: " << _con.min_rp << std::endl;
    std::cout << "max: " << _con.max_rp << std::endl;
    std::cout << "testing0: \n" << Xs.row(1) << std::endl;
    Eigen::VectorXd testing = Xs.row(1).cwiseMax(_con.min_rp);
    std::cout << "testing1:  \n" << testing.transpose() << std::endl;
    testing = testing.cwiseMin(_con.max_rp);

    std::cout << "testing2:  \n" << testing.transpose() << std::endl;
    Xs.row(0) = Xs.row(0).cwiseMax(_con.min_temp).cwiseMin(_con.max_temp);
    Xs.row(1) = Xs.row(1).cwiseMax(_con.min_rp).cwiseMin(_con.max_rp);
    Xs.row(2) = Xs.row(2).cwiseMax(_con.min_vp).cwiseMin(_con.max_vp);
    Xs.row(3) = Xs.row(3).cwiseMax(_con.min_uvi).cwiseMin(_con.max_uvi);
    Xs.row(4) = Xs.row(4).cwiseMax(_con.min_uvt).cwiseMin(_con.max_uvt);
    std::cout << "after _Xs: \n" << Xs << std::endl;
    
    // make copy of Xs in param
    _param_curr.block(0, 0, _m, _n) = Xs.transpose();

    // // evaluate objective function
    // #pragma omp parallel for 
    // for (int m = 0; m < _m; m++) {
    //   // simulation call
    //   Voxel sim(_sim.tfinal,                // tot sim time
    //             _sim.dt,                    // time step
    //             _sim.node,                  // num nodes
    //             _sim.time_stepping,         // sim id
    //             _param_curr(m, 0),          // amb temp
    //             _param_curr(m, 3),          // uv intensity
    //             _param_curr(m, 4),          // uv exposure time
    //             _file_path,
    //             _mthread);

    //   // generate particles
    //   sim.computeParticles(_param_curr(m, 1),
    //                        _param_curr(m, 2));

    //   // run simulation
    //   sim.simulate(_sim.method,             // time stepping scheme
    //                _save_voxel,             // save voxel values
    //                _obj_fn,                 // objective function
    //                _w                       // pareto weights
    //               );
    
    //   // record objective values
    //   #pragma omp critical
    //   {
    //     int thread_id = omp_get_thread_num();
    //     if (!std::isnan(sim.getObjective())) {
    //       _param_curr(m, 9) = sim.getObjective();
    //       _param_curr(m, 8) = sim.getObjM();
    //       _param_curr(m, 7) = sim.getObjMDot();
    //       _param_curr(m, 6) = sim.getObjPIDot();
    //       _param_curr(m, 5) = sim.getObjPI();
    //     } else {
    //       _param_curr(m, 9) = 1000.;
    //       _param_curr(m, 8) = 1000.;
    //       _param_curr(m, 7) = 1000.;
    //       _param_curr(m, 6) = 1000.;
    //       _param_curr(m, 5) = 1000.;
    //     }
    //   }
    // }

    // // rank candidate solutions
    // std::cout << "param: \n " << _param_curr << std::endl;
    // this->sort_data(_param_curr);

    // // copy ranked data back to Xs
    // Xs = _param_curr.block(0, 0, _m, _n).transpose();
    // std::cout << "Xs: \n" << Xs << std::endl;

    // // selection
    // inv_sigma = 1 / _sigma;
    // for (int i = 0; i < _m; ++i) {
    //   delta_xs.col(i) = (Xs.col(i) - mu) / inv_sigma; 
    // }

    // delta_ws.setZero();
    // for (int i = 0; i < _m_elite; ++i) {
    //   delta_ws += ws(i) * delta_xs.col(i);
    // }

    // // mean update
    // mu += _sigma * delta_ws;

    // // step size control
    // hsig = (int)(ps.norm() / std::sqrt(1. - std::pow(1. - cs, 2. * g))) < E * (1.4 + 2. / (_n + 1.));
    // ps   = (1. - cs) * ps + std::sqrt(cs * (2. - cs) * mu_eff) * L.inverse() * delta_ws;
    // _sigma *= std::exp((cs/ds) * (ps.norm()/E - 1.));

    // w0.setZero();
    // for (int i = 0; i < _m; ++i) {
    //   if (ws[i] >= 0) {
    //     w0[i] = ws[i];
    //   } else {
    //     w0[i] = _n * ws[i] / (L * delta_xs).squaredNorm();
    //   }
    // }

    // // update covariance matrix
    // C = (1. - c1 - cmu) * C                                              // regard old matrix
    //     + c1 * (pc * pc.transpose() + (1. - hsig) * cc * (2. - cc) * C)  // rank-one update
    //     + cmu * delta_ws * w0.transpose() * L.inverse() * delta_xs;      // adapt covariance

    // // enforce symmetry
    // C = (C + C.transpose()) * 0.5;

    // std::cout << "sigma: " << _sigma << std::endl;
    // std::cout << std::endl;

  }
}
// PRIVATE METHODS
void CMAES::init_Cov(Eigen::MatrixXd& C, Eigen::VectorXd& mu) {
  for (int i = 0; i < _n; ++i) {
    for (int j = 0; j < _n; ++j) {
      if (i == j) {
        C(i, j) = std::sqrt(  ((_max_constraints(i) - mu(i)) * (_max_constraints(j) - mu(j))
                            +  (_min_constraints(i) - mu(i)) * (_min_constraints(j) - mu(j)))
                              * 0.5
                            );
      } else {
        C(i, j) = std::sqrt(  ((_max_constraints(i) - mu(i)) * (_max_constraints(j) - mu(j))
                            +  (_min_constraints(i) - mu(i)) * (_min_constraints(j) - mu(j)))
                              * 0.5
                            );
        C(j, i) = C(i, j);

      }
    }
  }
}
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

void CMAES::stdz_data(Eigen::MatrixXd& param) {
    // check if data is already normalized
    if (_stdzd) { return; }
    // compute mean and stddev for param data
    _stdz_avg = param.block(0, 0, _m, _n).colwise().mean();
    _stdz_std = ((param.block(0, 0, _m, _n).rowwise() - _stdz_avg.transpose()).array().square().colwise().sum() / (_m - 1)).sqrt();

    // standardize the data
    for (int i = 0; i < _n; ++i) {
        param.col(i) = (param.col(i).array() - _stdz_avg(i)) / _stdz_std(i);
    }

    // set flag
    _stdzd = true;
}

void CMAES::unstdz_data(Eigen::MatrixXd& param) {
    // check if data is already unnormalized
    if (!_stdzd) { return; }

    // un-standardize data
    for (int i = 0; i < _n; ++i) {
        param.col(i) = (param.col(i).array() * _stdz_std(i)) + _stdz_avg(i);
    }

    // set flag
    _stdzd = false;
}
