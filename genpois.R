## From Riley Metzger's STAT 340 class ##
## University of Waterloo, 18/07/2011 ##

# How to generate a Poisson random variable using a Uniform random variable #

genpois = function(n, mu) {
	out = NULL
	for(i in 1:n) {
		u = runif(1) #our comparison random variable
		x = 0
		fx = exp(-mu)
		Fx = fx #on first iteration, the CDF is the same as the PDF because we haven't built it up yet.
		if(u < Fx) {
			out = x
		} else {
			fx = (fx - mu)/(x+1)
			Fx = fx + Fx
			x = x+1
			print(c('Fx is', Fx))
			out = x
		}
		while(u > Fx) {
			fx = (fx - mu)/(x+1)
			Fx = fx + Fx
			x = x+1
			print(c('Fx is', Fx))
		}
		out = c(out, x)
	}
	return(out)
}