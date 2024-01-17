// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#ifndef CMAES_H
#define CMAES_H

#include "common.h"
#include "Voxel.h"

class CMAES {

  private:
    // print/saving options
    bool _mthread;                             // flag for multithreading
    bool _save_voxel;                          // flag for saving voxel data


    // objective function
    int _obj_fn;                               // objective function
    double *_w;                                // weights for objective function
    size_t _w_size;                            // size of weights array

    std::string _file_path;

    // simulation parameters
    sim _sim;

    // optimization constraints
    constraints _con;

    // optimization parameters
    bopt _b;

    // CMA-ES parameters
    double _sigma;                              // step size
    int    _c;                                  // num of children
    int    _G;                                  // num of generations
    int    _m;                                  // pop size
    int    _m_elite;                            // num elites
    int    _n;                              // number of variables


    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    Eigen::MatrixXd _param_curr, _param_next;

    // performance vectors
    std::vector<double> _top_performer; 
    std::vector<double> _avg_parent; 
    std::vector<double> _avg_total;
    std::vector<double> _top_obj_pi, _top_obj_pidot, _top_obj_mdot, _top_obj_m;
    std::vector<double> _top_temp, _top_rp, _top_vp, _top_uvi, _top_uvt;

    // statistical parameters
    bool _stdzd;                                // flag for normalized data

    Eigen::VectorXd _max_constraints;           // max constraints        ∈ ℝ(n_var)
    Eigen::VectorXd _min_constraints;           // min constraints        ∈ ℝ(n_var)
    Eigen::VectorXd _stdz_avg;                  // param averages         ∈ ℝ(n_var)
    Eigen::VectorXd _stdz_std;                  // param stds             ∈ ℝ(n_var)
    Eigen::MatrixXd _Cov;                       // covariance matrix      ∈ ℝ(n_var x n_var)
    Eigen::MatrixXd _Xs;                        // stdzd, transposed data ∈ ℝ(n_var x m)
    Eigen::VectorXd _mu;                        // mean vector            ∈ ℝ(n_var)

    void init_Cov(Eigen::MatrixXd& Cov, Eigen::VectorXd& mu);

    void gen_Zs(Eigen::MatrixXd& Z);

    void sort_data(Eigen::MatrixXd& param);

    void stdz_data(Eigen::MatrixXd& param);

    void unstdz_data(Eigen::MatrixXd& param);

  public:
  
    /* overload constructor */
    CMAES(sim& s,
          constraints& c, 
          bopt& b, 
          int obj_fn,
          int n_var,
          const double* weights,
          std::string file_path);

    /* destructor */
    ~CMAES();

    /* initialize CMAES */
    void initialize();

    /* optimize CMAES */
    void optimize();
    
};

#endif // CMAES_H