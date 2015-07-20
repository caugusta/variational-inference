rm(list=ls())
#This is from Dec26-2.R

infcalc = function(pop, true_alpha, true_beta, tmax) {
	
		#These will be the infection times of each individual
		inf = rep(0, pop)

		#The coordinates of each individual on the grid
		x = rep(1:sqrt(pop), each=sqrt(pop))
		y = rep(1:sqrt(pop), sqrt(pop))	
		
		#Start the epidemic
		startinf = sample(1:pop, 1) 
		inf[startinf] = 1
		res1 = inf
		
		#Declaring coordinates of susceptible individuals
		xi = rep(0, length(inf))
		yi = rep(0, length(inf))
		
		for(Z in 2:tmax) { #for the rest of the observation time
			inf = res1
			probinf = sapply(1:length(inf), function(Y) {
				if(inf[Y]==0) {										#if individual is susceptible (not infectious and not removed)
					
					#Coordinates of susceptible individuals
					xi[which(inf==0)] = x[which(inf==0)]			#take the x coordinates of the susceptible individuals
					yi[which(inf==0)] = y[which(inf==0)]			#and their y coordinates					
					
					#Coordinates of infectious individuals
					xj = x[which(inf!=0)]
					yj = y[which(inf!=0)]
					
					#Keeping track of probabilities
					probinf2 = as.list(rep(0, length(inf)))
					
					#Building up d_ij^(-beta)
					a1 = outer(xi[Y], xj, "-")					#differences between x coordinates of susc and inf indivs
					b1 = outer(yi[Y], yj, "-")					#and their y coordinates
					dismat = (sqrt(a1^2 + b1^2))^(-true_beta)		#final matrix of the differences in distances between susc and inf indivs
					dis = rowSums(dismat) 						#distance between one susc indiv and all inf indivs (inf pressure)
					
					#probability of infection at time Z
					probinf1 = -expm1(-true_alpha*dis)
					print(c('probability of infection was', probinf1))
					#Assigning infection probabilistically			
					u = runif(1)
					if(u < probinf1) {
						probinf2[seq(1:length(xi))[Y]] = Z		#if we accept, then the indiv at position Y becomes inf at time Z
						inf = mapply("+", inf, probinf2)		#and the infectious population now includes Y.
						
					} else {
						probinf2[seq(1:length(xi))[Y]] = 0		#if we reject, make sure to keep at 0
					}
				} 
			})
			n1 = mapply(max, probinf, 0) #if all probinf == NULL, this is all 0: finds the maximum of probinf and 0, giving which are susc/rem
			if(any(n1 != 0)) {
				res1 = do.call(rbind, probinf)	#put the infection times in rows
				res1 = apply(res1, 2, max)	#get the max value of res1 in each column. This is the time of infection of the individual.
			}
		} 
		vec1 = cbind(res1, x, y)
	return(vec1)
	}