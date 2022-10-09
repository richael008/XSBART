predict.XSBART <- function(model, X) {
    
    out = json_to_r(model$tree_json)

    obj = .Call(`_XSBART_xbart_predict`, 1,X, model$model_list$y_mean, out$model_list$tree_pnt) # model$tree_pnt
    obj = as.matrix(obj$yhats)
    return(obj)
}


