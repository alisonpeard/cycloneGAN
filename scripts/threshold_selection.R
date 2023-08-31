library(reticulate)
library(threshr)
library(knitr)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

np <- import("numpy")
var <- 'precip_data'

X <- np$load(paste0("/Users/alison/Documents/DPhil/multivariate/", var, "/train/images.npy"))
M <- dim(X)[2]
N <- dim(X)[3]

u_mat <- matrix(nrow=M, ncol=N)  # matrix of threshold values
f_mat <- matrix(nrow=M, ncol=N)  # matrix of corresponding ECDF values
n.excesses <- matrix(nrow=M, ncol=N)

for(i in 1:M){
  for(j in 1:N){
    x <- X[, i, j, 1]
    if(var(x) > 0){
      npy <- 365 # observations for every day of the year
      attr(x, 'npy') <- npy
      
      # restrict search so that number of excesses always >= 50
      max.allowed <- sort(x)[(length(x) - 50)]
      max.quantile <- ecdf(x)(max.allowed)
      
      q_vec <- seq(.6, max.quantile, by=0.01)
      u_vec <- quantile(x, p=q_vec)
      
      suppressWarnings({
        var_cv <- ithresh(x, u_vec=u_vec)
      })
      
      best_u <- getmode(summary(var_cv)[, "best u"])
      #best_i <- getmode(summary(var_cv)[, "index of u_vec"])
      #best_q <- q_vec[best_i]
      best_f <- ecdf(x)(best_u)
      
      f_mat[i, j] <- best_f
      u_mat[i, j] <- best_u
      n.excesses[i, j] <- length(x[x > best_u])
    }else{
      f_mat[i, j] <- 0.
      u_mat[i, j] <- x[1]
    }
  }
}

np$save(paste0("/Users/alison/Documents/DPhil/multivariate/", var, "/train/thresholds.npy"), u_mat)
np$save(paste0("/Users/alison/Documents/DPhil/multivariate/", var, "/train/threshold_ecdfs.npy"), f_mat)



if(FALSE){
  summary(var_cv)
  pred <- predict(var_cv, which_u="best", n_years=1/12, type="d") # 1-month RP event distribution
  par(mfrow=c(2, 1))
  hist(x, breaks=50, probability=TRUE)
  abline(v=best_u, col='red', lty='dashed', lwd=2)
  hist(x[x>best_u], probability=TRUE, breaks=50)
  lines(pred$x, pred$y)
  abline(v=best_u, col='red', lty='dashed', lwd=2)
  mtext("Fitted Generalised Pareto for 10(?) years ", side = 3, line = 2.5)
  
}