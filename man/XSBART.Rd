\name{XSBART}
\alias{XSBART}
\title{XSBART: Accelerated Soft Bayesian Additive Regression Trees}
\description{
    Main function to fit XSBART model. Fit a regression tree (continuous output).
}
\usage{
XSBART(y, X, Xtest, num_trees, num_sweeps, max_depth=250, 
Nmin=1, num_cutpoints=100,alpha=0.95, beta=1.25, tau=NULL, 
no_split_penality = NULL, burnin = 1L, mtry = NULL, 
p_categorical = 0L, kap = 16, s = 4, tau_kap = 3, 
tau_s = 0.5, verbose = FALSE, sampling_tau = TRUE, 
parallel = TRUE, random_seed = NULL, sample_weights_flag = TRUE,
nthread = 0,ctType=1,RepeatT=6,sigma_hat=NULL,StepSize=40,
MHB = TRUE,NWB = FALSE,FitB=FALSE, ...)
}
\arguments{
  \item{y}{\code{matrix} of inputs \code{Y}. Response variable. }
  \item{X}{\code{matrix} of inputs \code{X}. Regressors. }
  \item{Xtest}{\code{matrix} of test set.}
  \item{num_trees}{Number of trees.}
  \item{num_sweeps}{Number of sweeps.}
  \item{max_depth}{\code{matrix}, \code{N_sweeps} by \code{M}. Indicate max depth of each tree in each sweep.}
  \item{Nmin}{Minimal node size.}
  \item{num_cutpoints}{For continuous variable, number of adaptive cutpoint candidates considered in each split.}
  \item{alpha}{Prior parameter of BART.}
  \item{beta}{Prior parameter of BART.}
  \item{tau}{Prior parameter of BART.}
  \item{no_split_penality}{Weight of no-split option. The default value is log(num_cutpoints), or you can take any other numbers (should be in log scale).}
  \item{burnin}{Number of burn-in sweeps.}
  \item{mtry}{Number of varaibles considered in each split. Like random forest.}
  \item{p_categorical}{Number of categorical regressors. (\code{X} and \code{Xtest}).}
  \item{kap}{Prior parameter of \eqn{\sigma}.}
  \item{s}{Prior parameter of \eqn{\sigma}.}
  \item{tau_kap}{Prior parameter of \eqn{\tau}.}
  \item{tau_s}{Prior parameter of \eqn{\tau}.}  
  \item{verbose}{If \code{TRUE}, print the progress on screen.}
  \item{sampling_tau}{If \code{TRUE},\eqn{\tau} will be sampled out for each iteration.}
  \item{parallel}{If \code{TRUE}, do computation in parallel.}
  \item{random_seed}{Random seed, should be a positive integer.}
  \item{sample_weights_flag}{If \code{TRUE},the split variable's selecting probability should be sample for each iteration.}  
  \item{nthread}{number of thread for parallel computation.}  
  \item{ctType}{1 linear 2 Sigmoid}
  \item{RepeatT}{How many grids we should search for optimal band width.}
  \item{sigma_hat}{initial \eqn{\sigma} estimation.}  
  \item{StepSize}{StepSize for the band width,StepSize/1000 is the percent of Sample Size.}
  \item{MHB}{Whether to turn on MH tree selection}
  \item{NWB}{discarded Hyper Parameter.}
  \item{FitB}{discarded Hyper Parameter.}
  \item{...}{optional parameters to be passed to the low level function \code{XBART}.}
}
\details{
  For details of the model, please see the original paper XBART: Accelerated Bayesian Additive Regression Trees.
}
\value{
  \item{loops}{A \code{vector} of number of elliptical slice sampler loops for each posterior sample.}
  \item{sigma}{A \code{vector} of posterior samples of residual standard error.}
  \item{vglobal}{A \code{vector} of posterior samples of the global shrinkage parameter.}
  \item{beta}{A \code{matrix} of posterior samples of coefficients.}
  \item{fitted.values}{Fitted values of the regression model. Take posterior mean of coefficients with 20\% burnin samples.}
  \item{residuals}{Residuals of the regression model, equals \code{y - fitted.values}.}
}
\author{ RAN HAO \email{ranhao@163.shufe.edu.cn} }
\keyword{XSBART}