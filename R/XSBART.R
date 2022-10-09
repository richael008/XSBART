GetSSigma <- function(X,Y) 
{
  
  stopifnot(is.matrix(X) | is.data.frame(X))
  
  if(is.data.frame(X)) {
    X <- model.matrix(~.-1, data = X)
  }
  
  
  fit <- cv.glmnet(x = X, y = Y)
  fitted <- predict(fit, X)
  sigma_hat <- sqrt(mean((fitted - Y)^2))
  # sigma_hat <- 0
  # if(nrow(X) > 2 * ncol(X)) {
  #   fit <- lm(Y ~ X)
  #   sigma_hat <- summary(fit)$sigma
  # } else {
  #   sigma_hat <- sd(Y)
  # }
  
  return(sigma_hat)
  
}

normalize_Sbart <- function(y) {
  a <- min(y)
  b <- max(y)
  y <- (y - a) / (b - a) - 0.5
  return(y)
}

unnormalize_Sbart <- function(z, a, b,c) {
  y <- (b - a) * (z +c+ 0.5) + a
  return(y)
}


XSBART <- function(y, X, Xtest, num_trees, num_sweeps, max_depth = 250, 
    Nmin = 1, num_cutpoints = 100, alpha = 0.95, beta = 1.25, tau = NULL, 
    no_split_penality = NULL, burnin = 1L, mtry = NULL, p_categorical = 0L, 
    kap = 16, s = 4, tau_kap = 3, tau_s = 0.5, verbose = FALSE, sampling_tau = TRUE, parallel = TRUE, random_seed = NULL, 
    sample_weights_flag = TRUE, nthread = 0,ctType=1,RepeatT=6,sigma_hat=NULL,StepSize=40,MHB = TRUE,NWB = FALSE,FitB=FALSE, ...) {
    
    if (!("matrix" %in% class(X))) {
        cat("Input X is not a matrix, try to convert type.\n")
        X = as.matrix(X)
    }
    
    if (!("matrix" %in% class(Xtest))) {
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest = as.matrix(Xtest)
    }
    
    if (!("matrix" %in% class(y))) {
        cat("Input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }
    
    if (dim(X)[1] != length(y)) {
        stop("Length of X must match length of y")
    }
    
    if (dim(X)[2] != dim(X)[2]) {
        stop("Column of X must match columns of Xtest")
    }
    
    if (is.null(random_seed)) {
        set_random_seed = FALSE
        random_seed = 0
    } else {
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed = TRUE
    }
    
    if (burnin >= num_sweeps) {
        stop("Burnin samples should be smaller than number of sweeps.\n")
    }
    
    if (is.null(no_split_penality) || no_split_penality == "Auto") {
        no_split_penality = log(num_cutpoints)
    }
    

    
    if (is.null(mtry)) {
        mtry = dim(X)[2]
        cat("mtry = p, use all variables. \n")
    }

    if (mtry > dim(X)[2]){
        mtry = dim(X)[2]
        cat("mtry cannot exceed p, set to mtry = p. \n")
    }

    if(p_categorical > dim(X)[2]){
        p_categorical = dim(X)[2]
        stop("p_categorical cannot exceed p")
    }
    # check input type
    

    
     
  
    check_non_negative_integer(burnin, "burnin")
    check_non_negative_integer(p_categorical, "p_categorical")
    
    check_positive_integer(max_depth, "max_depth")
    check_positive_integer(Nmin, "Nmin")
    check_positive_integer(num_sweeps, "num_sweeps")
    check_positive_integer(num_trees, "num_trees")
    check_positive_integer(num_cutpoints, "num_cutpoints")
    
    check_positive_integer(StepSize, "StepSize")

    
    check_scalar(no_split_penality, "no_split_penality")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")
    check_scalar(kap, "kap")
    check_scalar(s, "s")
    
    min_a <- min(y)
    max_b <- max(y)
    ychange<-normalize_Sbart(y)
    mean_c<-mean(ychange)
    ychange<-ychange-mean_c
    
    if(is.null(sigma_hat))
    {
      sigma_hat                          <- GetSSigma(X,ychange)
    }
    cat("Sigmahat ", sigma_hat, "\n")
    
    if(is.null(tau))
    {
      tau                          <- var(ychange) / num_trees
    }
    check_scalar(tau, "tau")
    #cat("MHB",MHB,"NWB",NWB,"\n")
    obj = XBART_cpp(ychange, X, Xtest, num_trees, num_sweeps, max_depth, 
        Nmin, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, 
        mtry, p_categorical, kap, s, tau_kap, tau_s, verbose, sampling_tau, parallel, set_random_seed, 
        random_seed, sample_weights_flag, nthread,ctType,RepeatT,sigma_hat,StepSize,MHB,NWB,FitB)

    obj$yhats_test <- unnormalize_Sbart(obj$yhats_test, min_a, max_b,mean_c)
    obj$sigma <- (max_b - min_a) * obj$sigma

    
    tree_json = r_to_json(mean(y), obj$model$tree_pnt)
    obj$tree_json = tree_json

    class(obj) = "XSBART"
    return(obj)
}