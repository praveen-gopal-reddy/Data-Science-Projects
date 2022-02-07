load(file="C:/praveen/leeds university/statistical theory and methods/coursework/killersandmotives.Rdata")
createsample(34)
####packages
install.packages("dplyr")
library("dplyr")
install.packages("psych")
library(psych)
install.packages("moments")
library(moments)
install.packages("ggplot2")
library(ggplot2)

##percentage of cleaning records
clean_percentage1=(nrow(mysample[mysample$AgeFirstKill=="99999",])/nrow(mysample))*100
clean_percentage2=(nrow(mysample[is.na(mysample$Motive),])/nrow(mysample))*100 
clean_percentage3=(nrow(mysample[mysample$YearBorn<"1900",])/nrow(mysample))*100


######cleaning mysample
mysample=mysample[!(is.na(mysample$Motive)),]
mysample=mysample[!(mysample$YearBorn<"1900"),]
mysample=mysample[!(mysample$AgeFirstKill=="99999"),]
mysample["CareerDuration"]=mysample[,3]-mysample[,2]



#######summarize numerically
a_flc=fortest_mysample[,c("AgeFirstKill","AgeLastKill","CareerDuration")]
sapply(a_flc, function(a_flc) c("Stand dev" = sd(a_flc), 
                                          "Mean"= mean(a_flc),
                                          "length" = length(a_flc),
                                          "Median" = median(a_flc),
                                          "skewness" = skewness(a_flc),
                                          "Minimum" = min(a_flc),
                                          "Maximum" = max(a_flc)
)

#######summarize graphically
hist(age_firstkill, 
     main="Age at first kill", 
     xlab="Age",
     ylab="density",
     border="blue", 
     col="green",
     xlim=c(0,80),
     las=1,
     density=20,
     prob=TRUE)

hist(age_lastkill, 
     main="Age at last kill", 
     xlab="Age",
     ylab="density",
     border="green", 
     col="blue",
     xlim=c(0,80),
     las=1,
     density=20,
     prob=TRUE)

hist(career_dur, 
     main="career duration", 
     xlab="years",
     ylab="density",
     border="brown", 
     col="yellow",
     xlim=c(0,50),
     las=1,
     density=20,
     prob=TRUE)

###correlation between age_Firstkill,age_lastkill and career_duration
cordata = mysample[,c(2,3,10)]     
head(cordata,n=5)
corr_variables <- round(cor(cordata), 1)
corr_variables

#correlation by ggplot 
ggplot(data=fortest_mysample)+geom_point(aes(x=age_firstkill,
                                             y=age_lastkill,color=as.factor(age_lastkill)),size=2,alpha=0.8)+
  #scale_x_continuous(name=age_firstkill,breaks=seq(0,100,5))+
  #scale_y_continuous(name=age_lastkill,breaks=seq(0,300,5))+
  ggtitle(label="relation between Age first kill and last kill")+
  labs(x="Age first kill",y="Age Last Kill")+
  theme_bw()

ggplot(data=fortest_mysample)+geom_point(aes(x=age_firstkill,
                                             y=career_dur,color=as.factor(career_dur)),size=2,alpha=0.8)+
  #scale_x_continuous(name=age_lastkill,breaks=seq(0,100,5))+
  #scale_y_continuous(name=motive,breaks=seq(0,300,5))+
  ggtitle(label="Relation between Age First kill and Career duration")+
  labs(x="Age first kill",y="Motive")+
  theme_bw()	 

  
  
par(mfrow = c(1, 2))
#######################estimation for mu and sigma for age first and last kill

mu_firstkill=mean(age_firstkill)
mu_lastkill=mean(age_lastkill)
sigma_firstkill=sd(age_firstkill)
sigma_lastkill=sd(age_lastkill)

avg_mean_firstkill  <- rep(NA, 200)
avg_mean_lastkill  <- rep(NA, 200)

#first kill mle mean
for(i in 1:200){
  
  x <- rnorm(n = 25, mean = mu_firstkill, sd = sigma_firstkill)
  avg_mean_firstkill[i] <- mean(x)
  }


hist(avg_mean_firstkill, xlim = c(0,40),xlab="sample mean",main="Estimating 
     mu parameter for Age at first kill")
abline(v = mu_firstkill, col = "red", lwd = 2)
abline(v = mean(avg_mean_firstkill), col = "blue", lty = 2, lwd = 2)

#lastkill mle mean
for(i in 1:200){
  
  x <- rnorm(n = 25, mean = mu_lastkill, sd = sigma_lastkill)
  avg_mean_lastkill[i] <- mean(x)
}
hist(avg_mean_lastkill, xlim = c(0,40),xlab="sample mean",main="Estimating 
     mu parameter for Age at last kill")
abline(v = mu_lastkill, col = "red", lwd = 2)
abline(v = mean(avg_mean_lastkill), col = "blue", lty = 2, lwd = 2)

#firstkill mle of sd
mu      <- mean(age_firstkill)
sigma   <- sd(age_firstkill) 


sigma2hat1 <- rep(NA, 200)
sigma2hat2 <- rep(NA, 200)

for(i in 1:200){
  
  x <- rnorm(n = 25, mean = mu, sd = sigma)
  
  sigma2hat1[i] <- sd(x)^2
  sigma2hat2[i] <- (24/25)*sd(x)^2   
}
hist(sigma2hat1, xlim = range(c(sigma2hat1, sigma2hat2)),xlab="variance",main="Estimating 
     variance parameter for Age at first kill")
abline(v = sigma^2, col = "green", lwd = 2)
abline(v = mean(sigma2hat1), col = "brown", lty = 2, lwd = 2)

hist(sigma2hat2, xlim = range(c(sigma2hat1, sigma2hat2)),xlab="variance",main="Estimating 
     variance parameter for Age at first kill")
abline(v = sigma^2, col = "green", lwd = 3)
abline(v = mean(sigma2hat2), col = "brown", lty = 2, lwd = 3)


###################career_duration mle of lambda
c_dur_mean=mean(career_dur)
c_dur_lamda_true=1/c_dur_mean
x    <-  rexp(n = 200, rate = c_dur_lamda_true)
xbar=mean(x)

loglik <- function(lambda){
  
  L <- (lambda^200)*exp(-lambda*200*xbar)
  return(log(L))
  
}
lambda <- (1:40)/100   

plot(lambda, loglik(lambda), xlab = "lambda", ylab = "log likelihood",
main="estimating lambda parameter for career duration",type = "l")
abline(v = 1/xbar, col = "pink") 
abline(h = loglik(1/xbar), col = "yellow")
abline(v=c_dur_lamda_true,col="green")


######Numerical summary of age at first kill for different types of motives.
motive_angel=c(fortest_mysample[fortest_mysample$Motive == "Angel of Death", "AgeFirstKill"])
motive_revenge=c(fortest_mysample[fortest_mysample$Motive == "Revenge or vigilante justice", "AgeFirstKill"])
motive_robbery=c(fortest_mysample[fortest_mysample$Motive == "Robbery or financial gain", "AgeFirstKill"])

describe(motive_angel)
describe(motive_revenge)
describe(motive_robbery)

############hisogram plot with density curve for age at first kill for different type of motives
par(mfrow = c(2, 2))
hist(motive_angel, 
     main="Angel of Death", 
     xlab="Age",
     ylab="density",
     border="blue", 
     col="green",
     xlim=c(0,80),
     las=1,
     density=20,
     prob=TRUE)
curve(dnorm(x, mean=age_fk_mean, sd=age_fk_sd), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")


hist(motive_robbery, 
     main="Robbery or financial gain", 
     xlab="Age",
     ylab="density",
     border="green", 
     col="blue",
     xlim=c(0,80),
     las=1,
     density=20,
     prob=TRUE)
curve(dnorm(x, mean=age_fk_mean, sd=age_fk_sd), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

hist(motive_revenge, 
     main="Revenge or vigilante justice", 
     xlab="Age",
     ylab="density",
     border="brown", 
     col="blue",
     xlim=c(0,80),
     las=1,
     density=20,
     prob=TRUE)
curve(dnorm(x, mean=age_fk_mean, sd=age_fk_sd), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")
	  
################Hypothesis testing calculation 
mean_angel=32.59
n_angel=22

mean_robbery=29.20
n_robbery=490

mean_revenge=30.09
n_revenge=57

mu=27
variance=74

### Z score
Z_angel = (mean_angel - mu)/sqrt(variance/n_angel)
Z_robbery = (mean_robbery - mu)/sqrt(variance/n_robbery)
Z_revenge = (mean_revenge - mu)/sqrt(variance/n_revenge)

####CI interval
CI_angel =  mean_angel + c(-1, 1)*1.96*sqrt(variance/n_angel) 
CI_robbery =  mean_robbery + c(-1, 1)*1.96*sqrt(variance/n_robbery) 
CI_revenge =  mean_revenge + c(-1, 1)*1.96*sqrt(variance/n_revenge) 

#####P-value
P_angel=2*pnorm(-abs(Z_angel))
P_robbery=2*pnorm(-abs(Z_robbery))
P_revenge=2*pnorm(-abs(Z_revenge))


###################Independent two samples calculations
#####pair1 angel of death and robbery motive
meandiff_pair1=3.39
s1=sd(motive_angel)
s2=sd(motive_robbery) 

t=qt(p = 0.975, df = 22 + 490 - 2)
sp=sqrt( ((22 - 1)*s1^2 + (490 - 1)*s2^2)/(490 + 22 - 2)  )
CI_pair1=meandiff_pair1 + c(-1, 1) * t * sp * sqrt(1/22 + 1/490)

#####pair2 angel death and revenge
meandiff_pair2=2.5
s1=sd(motive_angel)
s2=sd(motive_robbery) 

t=qt(p = 0.975, df = 22 + 57 - 2)
sp=sqrt( ((22 - 1)*s1^2 + (57 - 1)*s2^2)/(57 + 22 - 2)  )
CI_pair2=meandiff_pair1 + c(-1, 1) * t * sp * sqrt(1/22 + 1/57)

#####pair3 d robbery and revenge motive
meandiff_pair3=0.89
s1=sd(motive_robbery)
s2=sd(motive_revenge) 

t=qt(p = 0.975, df = 22 + 490 - 2)
sp=sqrt( ((490 - 1)*s1^2 + (57 - 1)*s2^2)/(490 + 57 - 2)  )
CI_pair3=meandiff_pair1 + c(-1, 1) * t * sp * sqrt(1/490 + 1/57)




