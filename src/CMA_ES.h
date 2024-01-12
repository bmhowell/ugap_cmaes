// Copyright 2023 Brian Howell
// MIT License
// Project: CMA-ES

#ifndef CMAES_H
#define CMAES_H

#include "common.h"
#include "Voxel.h"

class CMAES {

  private:
    // GA parameters
    int _pop;                                  // population size
    int _P;                                    // number of parents
    int _C;                                    // number of children
    int _G;                                    // number of generations
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
    constraints _c;

    // optimization parameters
    bopt _b;

    // CMA-ES parameters
    // double _sigma;                              // step size
    // int    _m;                                  // pop size
    // int    _m_eite;                             // num elites
    int    _n_var;                              // number of variables




    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    Eigen::MatrixXd _param_curr, _param_next;

    // performance vectors
    std::vector<double> _top_performer; 
    std::vector<double> _avg_parent; 
    std::vector<double> _avg_total;
    std::vector<double> _top_obj_pi, _top_obj_pidot, _top_obj_mdot, _top_obj_m;
    std::vector<double> _top_temp, _top_rp, _top_vp, _top_uvi, _top_uvt;

    // statistical parameters
    bool normed_data;                           // flag for normalized data
    Eigen::VectorXd _min_param;                 // min param for each var, used for normalization
    Eigen::VectorXd _max_param;                 // max param for each var, used for normalization
    Eigen::VectorXd _mu_curr;                   // mean vector for current generation
    Eigen::VectorXd _mu_next;                   // mean vector for next generation
    Eigen::MatrixXd _sigma;                     // covariance matrix

    void sort_data(Eigen::MatrixXd& param);

    void norm_data(Eigen::MatrixXd& param);

    void unnorm_data(Eigen::MatrixXd& param);

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