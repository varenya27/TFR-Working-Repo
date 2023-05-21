'''
make the python files that will run for sims
'''
import os 

noise =[0,1,2,3,4]
likelihood = [1,2]
for i in noise:
    for j in likelihood:
        os.system('@echo off')

        l = 'echo "likelihood__ = {}">1.py'.format(j)
        n='echo "noise__ = {}">>1.py'.format(i)
        cp = 'type 1.py checks.py > check_L{}_{}.py'.format(j,i)
        # print(l,n,cp)
        os.system(l)
        os.system(n)
        os.system(cp)
