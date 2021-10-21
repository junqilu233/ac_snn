import neuron6
import numpy as np

s_num=25
a_num=4
c_num=10

s_c=np.random.rand(s_num,c_num)*0.08+0.1
s_a=np.random.rand(s_num,a_num)*0.1+0.2

for i in range(5):
	s_a[i,0]=0
	s_a[20+i,1]=0
	s_a[i*5,2]=0
	s_a[i*5+4,3]=0
	
for i in range(c_num):
	s_c[12,i]=0

f=open('parameters7.txt',mode='w')
for i in range(s_num):
	for j in range(a_num):
		f.writelines(f'{s_a[i,j]}\n')
for i in range(s_num):
	for j in range(c_num):
		f.writelines(f'{s_c[i,j]}\n')
f.close()

