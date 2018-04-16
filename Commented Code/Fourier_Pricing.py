#Import required libraries
from __future__ import division, print_function
import scipy.optimize as optimize
import numpy as np
import cmath as cm
import mpmath as mp
import math
import csv
import pandas as pd
import sys

#Function to return characteristic function
def char_fun(u, mu, sigma, lemu, lemd, etau, etad,T,S0):
    lem =lemu +lemd #Define net arrival rate for up and down jumps
    p = lemu/lem #Conditional probability of an up jump given that a jump is happening
    charac_exp = -0.5*(sigma**2)*(u**2) + mp.mpc(0,1)*mu*u + mp.mpc(0,1)*u*lem*(mp.fdiv(p,etau-mp.mpc(0,1)*u)-mp.fdiv(1-p,etad+mp.mpc(0,1)*u)) #Characteristic expression
    phiT = mp.exp(mp.mpc(0,1)*u*mp.log(S0))*mp.exp(T*charac_exp) #Characteristic function
    return phiT

#Function to calculate price of a call option
def call_price_calc(given,op):
    #Option data
    K = given[0] #Strike
    R = given[1] #Risk free rate of return (in %)
    T = given[2] #Time to expiry
    y = given[3] #Annual dividend yield
    S0 = given[4] #Stock price
    #Parameters over which data is being optimized
    sigma = op[0] #Log return diffusion
    lemu = op[1] #Arrival rate of up jumps
    lemd = op[2] #Arrival rate of down jumps
    etau = op[3] #Strength of up jump
    etad = op[4] #Strength of down jump
    k = math.log(K) #Log of strike rate
    r = R/100 #Risk free rate of return
    mu = r-y- lemu/(etau-1) + lemd/(etad +1) #Mu for the given data and parameters
    #Risk neutral probability of finishing in the money
    pi2 = 0.5 + mp.fdiv(1,mp.pi)*mp.quad(lambda u: mp.re(mp.fdiv(mp.exp(-mp.mpc(0,1)*u*k)*char_fun(u,mu,sigma,lemu,lemd,etau,etad,T,S0),mp.mpc(0,1)*u)),[0,mp.inf]) 
    #Delta of the option
    pi1 = 0.5 + mp.fdiv(1,mp.pi)*mp.quad(lambda u: mp.re(mp.fdiv(mp.exp(-mp.mpc(0,1)*u*k)*char_fun(u-mp.mpc(0,1),mu,sigma,lemu,lemd,etau,etad,T,S0),mp.mpc(0,1)*u*char_fun(-mp.mpc(0,1),mu,sigma,lemu,lemd,etau,etad,T,S0))),[0,mp.inf])
    #Price of the  call option
    C_alternate = S0*mp.exp(-y*T)*pi1 - K*mp.exp(-r*T)*pi2
    return C_alternate  

#Function to calculate price of a put option
def put_price_calc(given,op):
    #Option data
    K = given[0] #Strike
    R = given[1] #Risk free rate of return (in %)
    T = given[2] #Time to expiry
    y = given[3] #Annual dividend yield
    S0 = given[4] #Stock price
    #Parameters over which data is being optimized
    sigma = op[0] #Log return diffusion
    lemu = op[1] #Arrival rate of up jumps
    lemd = op[2] #Arrival rate of down jumps
    etau = op[3] #Strength of up jump
    etad = op[4] #Strength of down jump
    k = math.log(K) #Log of strike rate
    r = R/100  #Risk free rate of return
    mu = r-y- lemu/(etau-1) + lemd/(etad +1)#Mu for the given data and parameters
    #Risk neutral probability of finishing in the money
    pi2 = 0.5 + mp.fdiv(1,mp.pi)*mp.quad(lambda u: mp.re(mp.fdiv(mp.exp(-mp.mpc(0,1)*u*k)*char_fun(u,mu,sigma,lemu,lemd,etau,etad,T,S0),mp.mpc(0,1)*u)),[0,mp.inf])
    #Delta of the option
    pi1 = 0.5 + mp.fdiv(1,mp.pi)*mp.quad(lambda u: mp.re(mp.fdiv(mp.exp(-mp.mpc(0,1)*u*k)*char_fun(u-mp.mpc(0,1),mu,sigma,lemu,lemd,etau,etad,T,S0),mp.mpc(0,1)*u*char_fun(-mp.mpc(0,1),mu,sigma,lemu,lemd,etau,etad,T,S0))),[0,mp.inf])
    #Price of the put option
    P_alternate = K*mp.exp(-r*T)*(1-pi2) - S0*mp.exp(-y*T)*(1-pi1)
    return P_alternate 

#Function to calculate least squared criterion normalized by eucleidean distance
def optimizer(sigma, lemu, lemd, etau, etad):
    #Parameters received from MLE optimization
    orig = [sigma, lemu, lemd, etau, etad]
    op = orig #Array for paramters which will be optimized
    xl = pd.ExcelFile('final_copy.xlsx') #Read option data file
    df = xl.parse("final copy") #Parse option data file
    #Pull option from parsed data file
    K = df.iloc[:,1].tolist() #Strike
    opt_type = df.iloc[:,2].tolist() #Option type: Call or Put
    price = df.iloc[:,8].tolist() #Option price
    S0 = df.iloc[:,9].tolist() #Stock price
    R = df.iloc[:,10].tolist() #Risk free rate of return (in %)
    T_old = df.iloc[:,11].tolist() #Time to expiry (in days)
    T = [x/252 for x in T_old] #Time to expiry (in years)
    y_old = df.iloc[:,13].tolist() #Annual dividend yield (in %)
    y = [x/100 for x in y_old] #Annual dividend yield
    price_calc = [] #List to save calculated prices
    f = open('Fourier_Pricing.txt','w') #File handle to save calculated prices in txt
    for i in range(0,len(S0)): #Iterate over all options
        given = [K[i], R[i], T[i], y[i], S0[i]] #Option data
        if opt_type[i] == "call": #Check if call
            temp = call_price_calc(given,op) #Calculate call price for the option
            price_calc.append(temp) #Append to price list
            f.write(call_price_calc(given,op)) #Print price to txt
        else: 
            temp = put_price_calc(given,op) #Calculate put price for the option
            price_calc.append(temp) #Append to price list
            f.write(temp) #Print price to txt
    #Create dataframe from option price and option types
    df = pd.DataFrame({'Calculated Price':price_calc,'Option Type':opt_type}) 
    #Write dataframe to excel file
    df.to_excel('Calculated_Price.xlsx',sheet_name='sheet1',index='False')
    f.close() #Close write object


#Main function indicator
if __name__ == "__main__":
    #Call optimizer with MLE optimized parameter
    optimizer(0.212, 0.05, 0.09, 29.99, 19.37)
    
