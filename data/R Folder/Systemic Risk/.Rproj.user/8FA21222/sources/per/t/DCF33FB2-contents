library(systemicrisk)
"Based on work from: https://csh.ac.at/vis/code/network_optimization/"
rm(list=ls())
path_out_L <- "C:/"
path_out_E <- "C:/"
# set.seed(50)

# n_b <- 2; n_m <- 3; n_s <- 25 # 30
# n_b <- 1; n_m <- 2; n_s <- 7 # 10
# n_b <- 2; n_m <- 3; n_s <- 15 # 20

n_b <- 10 # 100 for the uniform


#########################################
# For N=100 leverage experiment
ab_l <- 9000 # 9000
ab_u <- 10000


lb_l <- 1000
lb_u <- 2000



gu_scale <- 1.0 
gl_scale <- 1.0 

a <- runif(n_b,gl_scale*ab_l,gu_scale*ab_u)
l <- a + rnorm(n_b,lb_l,lb_u)

print(a)
print(l)
n <- length(a)
a[n+1] <- if(sum(l)-sum(a) >0){sum(l)-sum(a)}else{0}
l[n+1] <- if(sum(a)-sum(l) >0){sum(a)-sum(l)}else{0}
n <- length(a)
E <- (a+l)/n # This should be n?

# use the systemicrisk package
mod <- Model.additivelink.exponential.fitness(n,alpha=-2.5,beta=0.4,gamma=0.8,
                                              lambdaprior=Model.fitness.genlambdaparprior(ratescale=20))

thin <-100 

print(sample_HierarchicalModel(l=l, a=a, model=mod, nsamples=10000, thin=thin, silent=TRUE))
res <- sample_HierarchicalModel(l=l, a=a, model=mod, nsamples=10000, thin=thin, silent=TRUE)

# rounding the variables to two digits (monetary units)
count <- 0
for (L in res$L){
  # print(L)
  count <- count + 1
  L_file_n = paste("R_liability_network", count, sep="_")
  write.csv(L, file.path(path_out_L, paste(L_file_n, ".csv", sep="")))
}

E <- round(E,2)
E_file_n <- "R_networth_values"
write.csv(E, file.path(path_out_E, paste(E_file_n, ".csv", sep="")))
print("DONE")

  