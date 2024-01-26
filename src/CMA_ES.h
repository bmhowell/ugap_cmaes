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
    int    _top_evals;                          // num of evals per gen
    int    _m;                                  // cmaes pop size
    int    _m_elite;                            // cmaes/ga num parents
    int    _c;                                  // ga num of children
    int    _G;                                  // num of generations
    int    _n;                                  // number of variables


    // || temp | rm | vp | uvi | uvt | obj_pi | obj_pidot | obj_mdot | obj_m | obj || ∈ ℝ (pop x param + obj)
    Eigen::MatrixXd _param_curr;

    // performance vectors
    std::vector<double> _top_performer, _avg_parent, _avg_total;
    std::vector<double> _top_obj_pi, _top_obj_pidot, _top_obj_mdot, _top_obj_m;
    std::vector<double> _top_temp, _top_rp, _top_vp, _top_uvi, _top_uvt;

    Eigen::VectorXd _max_constraints;           // max constraints        ∈ ℝ(n_var)
    Eigen::VectorXd _min_constraints;           // min constraints        ∈ ℝ(n_var)
    std::vector<double> _objs;                  // objective function     ∈ ℝ(pop)

    // temp matrices for sorting
    Eigen::MatrixXd _param_curr_temp;
    Eigen::MatrixXd _Xc_temp;
    Eigen::MatrixXd _Zs_temp;

    void gen_Zs(Eigen::MatrixXd& Z);

    void sort_param(Eigen::MatrixXd& PARAM);
    
    std::vector<size_t> sort_objs(std::vector<double>& OBJS);

    void run_par_sim();

    void track_var();

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

    /* optimize CMAES */
    void optimize();

    
};

#endif // CMAES_H