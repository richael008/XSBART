#ifndef GUARD_fit_info_h
#define GUARD_fit_info_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>
#include <RcppArmadillo.h>

class State
{
public:
    // residual size
    size_t dim_residual;

    // vectors (slop?)
    matrix<double> residual_std;
    matrix<double> treeest_std;
    std::vector<double> full_residual;
    //arma::mat TimeTable;
    time_t StartTime;
    time_t EndTime;

    // Random
    std::vector<double> prob;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> d;

    // Splits
    matrix<double> split_count_all_tree;
    std::vector<double> split_count_all;
    std::vector<double> split_count_current_tree;
    std::vector<double> mtry_weight_current_tree;

    // mtry
    bool use_all = true;
    bool parallel = true;
    bool MH_B;
    bool NW_B;
    bool Fit_B;

    // fitinfo
    size_t n_min;
    size_t n_cutpoints;
    size_t p_categorical;
    size_t p_continuous;
    size_t p; // total number of variables = p_categorical + p_continuous
    size_t mtry;
    size_t n_y;                       // number of total data points in root node
    const double *X_std;              // pointer to original data
    std::vector<double> *y_std; // pointer to y data
    size_t max_depth;
    size_t num_trees;
    size_t num_sweeps;
    size_t burnin;
    size_t ctType;
    size_t RepeatT;
    size_t StepSize;


    bool sample_weights_flag;
    double ini_var_yhat;
    double width_rate;

    matrix<size_t> Xorder_std;

    // residual standard deviation
    double sigma;
    double sigma2; // sigma squared
    double sigma_ini;
    

    // paralization
    size_t nthread;

    // Logit Model
    // lambdas
    std::vector<std::vector<std::vector<double>>> lambdas;
    std::vector<std::vector<std::vector<double>>> lambdas_separate;

    void update_sigma(double sigma)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        return;
    }
    
    State(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights_flag, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread,size_t ctType,size_t RepeatT,size_t StepSize,bool MHB,bool NWB,bool FitB)
    {

        // Init containers
        // initialize predictions_std at given value / number of trees
        // ini_xinfo(this->predictions_std, N, num_trees, ini_var_yhat / (double)num_trees);

        // initialize yhat at given value

        // this->residual_std = std::vector<double>(N);
        // this->residual_std_full = std::vector<double>(N);

        // Warning! ini_matrix(matrix, N, p).
        ini_matrix(this->residual_std, N, dim_residual);
        ini_matrix(this->treeest_std,N,num_trees);

        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        ini_xinfo(this->split_count_all_tree, p, num_trees);

        this->split_count_current_tree = std::vector<double>(p, 0);
        this->mtry_weight_current_tree = std::vector<double>(p, 0);
        this->split_count_all = std::vector<double>(p, 0);
        this->full_residual=std::vector<double>(N, 0);
        this->sigma = sigma;
        
        this->n_min = n_min;
        this->n_cutpoints = n_cutpoints;
        this->p_categorical = p_categorical;
        this->p_continuous = p_continuous;
        this->mtry = mtry;
        this->X_std = X_std;
        this->p = p_categorical + p_continuous;
        this->n_y = N;
        this->num_trees = num_trees;
        this->num_sweeps = num_sweeps;
        this->sample_weights_flag = sample_weights_flag;
        this->y_std = y_std;
        this->max_depth = max_depth;
        this->width_rate=10;
        this->burnin = burnin;
        this->ini_var_yhat = ini_var_yhat;
        this->Xorder_std = Xorder_std;
        //this->TimeTable = arma::zeros<arma::mat>(num_trees*num_sweeps+2, 8);
        
        this->nthread = nthread;
        this->ctType=ctType;
        this->NW_B=NWB;
        this->MH_B=MHB;
        this->Fit_B=FitB;
        if (cStep<RepeatT) 
        {
            this->RepeatT=cStep;  
        }
        else
        {
            this->RepeatT=RepeatT;  
        }

        if (this->RepeatT*StepSize>500)
        {
            this->StepSize = 500/this->RepeatT;  
        }
        else
        {
            this->StepSize = StepSize;
        }   
            
        return;
    }


    void update_split_counts(size_t tree_ind)
    {
        mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;
        split_count_all_tree[tree_ind] = split_count_current_tree;
        return;
    }
    // void RecordTime(size_t i,size_t j )
    // {
    //     double ResTime;
    //     this->EndTime=clock();
    //     ResTime=double(this->EndTime-this->StartTime)/CLOCKS_PER_SEC;
    //     //
    //     this->TimeTable(i,j)=ResTime;
    //     this->StartTime=this->EndTime;
    //     return ;
    // }

    // void RecordData(size_t i,size_t j,double d)
    // {
    //     this->TimeTable(i,j)=d;
    //     return ;
    // }

    // void RecordAvg(size_t i,size_t j)
    // {
    //     double ResTime;
    //     this->EndTime=clock();
    //     ResTime=double(this->EndTime-this->StartTime);
    //     this->TimeTable(i,j)=this->TimeTable(i,j)+ResTime;
    //     this->TimeTable(i,j+1)+=1.0;
    //     this->StartTime=this->EndTime;
    //     return ;
    // }





};

class NormalState : public State
{
public:


    NormalState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights_flag, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel,size_t ctType,size_t RepeatT,size_t StepSize,bool MHB,bool NWB,bool FitB) : State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, n_cutpoints, mtry, X_std, num_sweeps, sample_weights_flag, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread,ctType,RepeatT,StepSize,MHB,NWB,FitB)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        this->parallel = parallel;
        this->sigma_ini=sigma;
    }


};

#endif
