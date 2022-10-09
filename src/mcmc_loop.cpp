#include "mcmc_loop.h"
#include "omp.h"



void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct,arma::mat &PerMatrix_std)
{

    // if (state->parallel)
    //     thread_pool.start();

    // Residual for 0th tree
    // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];

    
    model->ini_residual_std(state);
    //state->RecordTime(0,2);


    //ofstream outputfile; 
    //outputfile.open ("log-uw.csv",ios::app);
    //outputfile<<"sweeps,tree_ind,total Node,node,v,c,update,fper,fwidth,nper,nwidth,newloglik,oldloglik"<<std::endl;
    //outputfile.close();

    // ofstream outputfile;             
    // outputfile.open ("log-CK.csv",ios::app);            
    // outputfile<<"E,"<<state->MH_B<<","<<state->NW_B<<std::endl;
    // outputfile.close();    

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw Sigma
            
            model->update_state(state, tree_ind, x_struct);
            //cout << "UPDATE sigma" << state->sigma << endl;

            sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights_flag)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);
            //state->RecordTime(1+sweeps*state->num_trees+tree_ind,3); 

            if (state->parallel)
            {
                // run parallel
                // omp_set_max_active_levels(state->nthread);
// #pragma omp parallel default(none) shared(trees, sweeps, state, Xorder_std, x_struct, model, tree_ind)
                // {
// #pragma omp sections
                    // {
// #pragma omp section
                        // {
                trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
                        // }
                    // }
                // }
            }
            else
            {
                // single core
                trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
            }
            //state->RecordTime(1+sweeps*state->num_trees+tree_ind,4); 
            //if ((tree_ind==state->num_trees-1) && (sweeps == state->num_sweeps-1))
            //{
            
            // tree::npv mnods;
            // trees[sweeps][tree_ind].getnodes(mnods);
            // for(size_t i1=0;i1<mnods.size();i1++)
            // {
            //     mnods[i1]->fWidth=0;
            // }            
            // double CK1=trees[sweeps][tree_ind].LogLT(state,model);

            // ofstream outputfile; 
            // outputfile.open ("log-CK.csv",ios::app);
            //state->RecordTime(1+sweeps*state->num_trees+tree_ind,2); 
            if (state->Fit_B)
            {
                tree::npv bnogs;
                trees[sweeps][tree_ind].getnogs(bnogs);
                for(size_t i1=0;i1<bnogs.size();i1++)
                {
                    if (bnogs[i1]->maxv<0) 
                    {
                        ofstream outputfile; 
                        outputfile.open ("log-ERR.csv",ios::app);
                        outputfile<<"WrongType,Invalid Best Choice"<<"Sweeps,"<<sweeps<<"tree_ind,"<<tree_ind<<std::endl;
                        outputfile.close();
                    } 
                    else
                    {
                        bnogs[i1]->setv(bnogs[i1]->maxv);
                        bnogs[i1]->setc(bnogs[i1]->maxc);
                    }
                    
                    
                    // outputfile<<"B,"<<bnogs[i1]->nid()<<","<<bnogs[i1]->getv()<<","<<bnogs[i1]->maxv<<","<<bnogs[i1]->getc()<<","<<bnogs[i1]->maxc<<std::endl;

                }
            }

            // if (bnogs.size()>=1)
            // {   
            //     tree::npv nds;
            //     trees[sweeps][tree_ind].getnodes(nds);
            //     for(size_t i1=0;i1<nds.size();i1++)
            //     {
            //         outputfile<<"N,"<<nds[i1]->nid()<<","<<nds[i1]->getv()<<","<<nds[i1]->getc()<<std::endl;
            //     }

            // }
            // double CK2=trees[sweeps][tree_ind].LogLT(state,model);
            // outputfile<< "CK," << CK1<<","<<CK2<<std::endl;
            // outputfile.close();

                    
             

            double loglik_new;
            //cout << "UPDATE width-S" << endl;
            //state->RecordTime(1+sweeps*state->num_trees+tree_ind,1);
            loglik_new=trees[sweeps][tree_ind].updateWidth(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind,PerMatrix_std );
            //cout << "UPDATE width-E" << endl;
            //}
            //state->RecordTime(1+sweeps*state->num_trees+tree_ind,7);
             
            if (sweeps>0 && state->MH_B)
            {

                double loglik_old=trees[sweeps-1][tree_ind].LogLT(state,model);
                //double new_split=0;
                //double old_split=0;
                
                double new_split= trees[sweeps][tree_ind].prior_prob(model);
                double old_split= trees[sweeps-1][tree_ind].prior_prob(model);
                
                double cutoff = loglik_new + new_split - loglik_old - old_split;

                //ofstream outputfile; 
                //outputfile.open ("log-MH.csv",ios::app);
                //outputfile<<sweeps<<","<<tree_ind<<","<<loglik_new<<","<<new_split<<","<<loglik_old<<","<<old_split<<","<<cutoff<<",";
                



                if (log(unif_rand()) >= cutoff)
                {

                    //outputfile<<"Change back"<<std::endl;
                    trees[sweeps][tree_ind].tonull();
                    trees[sweeps][tree_ind].cp(&(trees[sweeps][tree_ind]),&(trees[sweeps-1][tree_ind]));
                                       

                }
                else
                {
                    //outputfile<<"Change,"<<std::endl;

   
                    

                } 
                //outputfile.close();
               
            }







            trees[sweeps][tree_ind].updateMu(state, model); 

            
            arma::vec predv = arma::zeros<arma::vec>(state->n_y);  
            trees[sweeps][tree_ind].predict(state,predv);  


            // update tau after sampling the tree
            // model->update_tau(state, tree_ind, sweeps, trees);
            Vec_To_Matrix(predv,state->treeest_std,tree_ind);

            state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            model->state_sweepU(state,tree_ind, state->num_trees, state->residual_std,state->full_residual, x_struct);

            //ofstream outputfile;
            //outputfile.open ("residual.csv",ios::app);
            //outputfile<<sweeps<<"," << tree_ind<<"," <<sum_squared(state->full_residual)<<"," <<std::endl; 
            //outputfile.close();
             
        }

        //cout << "UPDATE tau-S" << endl;
        
        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees);
            //state->RecordTime(1+sweeps*state->num_trees+state->num_trees-1,7);             
        }
        //cout << "UPDATE tau-E" << endl;
    }
    // thread_pool.stop();


    return;
}
