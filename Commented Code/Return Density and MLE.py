# Import required Libraries
from __future__ import division,print_function
import scipy.integrate as integrate
import scipy.optimize as optimize
import numpy as np
import sys
import pandas as pd
from numpy import inf
from math import exp,factorial,sqrt,pi

# Function to calculate maximum likelihood estimation
def mlecode():
    xl = pd.ExcelFile('spx_12.xlsx') #Read stock data file
    df = xl.parse("Sheet1") #Parse stock data file
    r = df.iloc[:,0].tolist() #Create a list of stock prices for SPX over 12 years
    x0 = [0.0005, 0.008 ,0.01, 0.01, 0.01, 0.01] # Initial guesses for the parameters
    # Mu, Sigma, Etau, Etad, Lambdau, Lambdad
    fl = open('out.txt','w') #Create object to print result to file
    non_neg = 10**(-10) #Define >0
    bnd =[(None,None), (non_neg, None),(non_neg, None),(non_neg, None),(non_neg, None),(non_neg, None)] #Define bounds for parameters
    print("entering optimization\n") #Optimization indicator
    sys.stdout.flush() #Flush buffer to allow printing before program is finished interpreting 
    estimates = optimize.minimize(fun,x0,args = (r,fl), method ='L-BFGS-B',bounds =bnd,options={'disp':True}) #Minimization using L-BFGS-B algorithm
    #over the defined bounds
    fl.close() #Close file object
    print(estimates) #Print estimates MLE

#Function for which MLE is calculated  
def fun(x,rs,fl): 
    a = 0
    for r in rs: #Iterate over stock prices
        mix = mixturedensity(r,x,fl) #Calculate mixture density for given parameters for each stock price
        if (mix != 0):
            a = a + np.log(mix) #Calculate log of mixture density and add it to the existing sum
        else:
            a = a+ np.log(10**(-10)) #Add a very negative value for mixture density = 0 
    return -a #Return negative log mixture density for all stock prices

#Function to calculate mixture density using Ramezeni (1999) paper, referred in main text
def mixturedensity(r,x,fl):
   #Initialize subfunctions used to calculate mixture density  
   f1 = 0
   f2 = 0
   f3 = 0
   f4 =0
   f = 0
   sys.stdout.flush()
   #Save parameters from argument 
   mu = x[0]
   sigma = x[1]   
   s = 1
   etau = x[2]
   etad = x[3]
   lemu = x[4]
   lemd = x[5]
   non_neg = 10**(-10) #Define >0
   fl.write(str(x) +"\n") #Write parameters to file object
   f1 = exp(-(lemu + lemd))*f00(r,s,sigma,mu) #Calculate subfunction f1
   #Perform summation from n=1 to upperbound 100 to calculate subfunction f2
   n= 1
   while( n<100):
        x =exp(-lemu)*P(n,lemd)*f0n(n,s,r,mu,sigma,etad) #Discrete calculation for a value of n
        temp = f2
        f2 = f2 + x #Summation action
        #Break condition using the required accuracy
        if(2*abs(x) <= (abs(f2) + abs(temp))*non_neg):
            break
        n = n+1
   #Perform summation from m=1 to upperbound 100 to calculate subfunction f3     
   m =1
   while (m<100):
        x= exp(-lemd)*P(m,lemu)*fm0(m,s,r,mu,sigma,etau) #Discrete calculation for a value of m
        temp = f3
        f3 = f3 +x #Summation action
        #Break condition using the required accuracy
        if (2*abs(x) <= (abs(f3) + abs(temp))*non_neg):
            break

        m = m+1
   #Perform double summation over m and n using lowerbounds 1 and upperbounds 100 to calculate sunfunction f4
   breakcounter=0     
   m= 1
   n =1
   while(n <20):
        while(m <20):
            x = P(n,lemd)*P(m,lemu)*fnm(r,etau,etad,m,n,s,sigma,mu); #Discrete calculation for a value of m and n
            temp = f4
            f4 = f4 +x
            m = m +1
            #Break condition using the required accuracy
            if (2*abs(x) <= (abs(f4) + abs(temp))*non_neg):
                breakcounter=1
                break
            n = n+1
        #Break n loop if m loop is broken    
        if (breakcounter == 1):
            break
   f = f1 + f2 + f3+ f4 # Calculate final mixture density
   return f

#Calculate conditional density of (0,0) up and down jumps f00 used within the mixture density subfunction calculations
def f00(r,s,sigma,mu): 
    a = (float(1)/(sigma*sqrt(2*pi*s)))*exp(-((r-mu*s+0.5*(sigma**2)*s)**2)/(2*(sigma**2)*s)) #Calculate f00 value
    return a

#Calculate conditional density of (0,n) up and down jumps f0n used within the mixture density subfunction calculations
def f0n(n,s,r,mu,sigma,etad):
    fun = lambda x: ((-x)**(n-1))*exp(etad*x-((r-x-mu*s+0.5*(sigma**2)*s)**2)/float((2*(sigma**2)*s))) #Define integrand
    ca =(float(etad**n)/(factorial(n-1)*sigma*sqrt(2*pi*s))) #Define constant outside integral
    a =integrate.quad(fun,-2,0) #Integrate with limits -2,0
    a = ca*a[0] #Calculate f0n value 
    return a

#Calculate conditional density of (m,0) up and down jumps fm0 used within the mixture density subfunction calculations   
def fm0(m,s,r,mu,sigma,etau):
    fun = lambda x: (x**(m-1))*exp(-etau*x-((r-x-mu*s+0.5*(sigma**2)*s)**2)/float((2*(sigma**2)*s))) #Define integrand
    ca =(float(etau**m)/(factorial(m-1)*sigma*sqrt(2*pi*s))) #Define constant outside integral
    a = integrate.quad(fun,0,inf) #Integrate with limits 0,Inf
    a = ca*a[0] #Calculate fm0 value
    return a

#Calculate density of Poisson random variable used within the mixture density subfunction calculations    
def P(n,lam):
    a = ((lam**n)*(exp(-lam)))/float(factorial(n)) 
    return a

#Calculate conditional density of (m,n) up and down jumps fnm used within the mixture density subfunction calculations
def fnm(r,etau,etad,m,n,s,sigma,mun):
    sm = 0
    con = float((etau**m)*(etad**n))/(factorial(m-1)*factorial(n-1)*sigma*sqrt(2*pi*s)) #Define constant outside integral
    itr = np.linspace(-2,100,.01) #Define integral bounds -2,100 
    #with step size 0.01 because internal integral upper bound is depedent on dummy variable (0^i)
    sm = 0;
    for i in itr: #Internal integral with upper bound (0^i) and lower bound -2
        if(i <0): #For i<0, 0^i is i
            g = integrate.quad(lambda x: ((-x)**(n-1))*((i-x)**(m-1))*exp((etau+etad)*x),-2,i)
        else: #For i>=0, 0^i is 0
            g = integrate.quad(lambda x: ((-x)**(n-1))*((i-x)**(m-1))*exp((etau+etad)*x),-2,0)
        #Outer integral calculate from i to i+step size
        f = integrate.quad(lambda t: g[0]*exp(-etau*t-((r-t-mun*s+0.5*(sigma**2)*s)**2)/float((2*(sigma**2)*s))),i,i+0.01) 
        sm = sm + f[0] #Combine integral from -2 to 100
    a = con*sm #Calculate fnm value
    return a
 

# Main function indicator    
if __name__ == "__main__":
    mlecode()

