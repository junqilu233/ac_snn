import neuron6
import numpy as np
import random
import matplotlib.pyplot as plt

state=[]
action=[]
critic=[]

s_num=25
a_num=4
c_num=10

value=[0.6561,0.729,0.81,0.729,0.6561,
	   0.729,0.81,0.9,0.81,0.729,
	   0.81,0.9,0,0.9,0.81,
	   0.729,0.81,0.9,0.81,0.729,
	   0.6561,0.729,0.81,0.729,0.6561]

for i in range(s_num):
	state.append(neuron6.neuron())
for i in range(a_num):
	action.append(neuron6.neuron())
for i in range(c_num):
	critic.append(neuron6.neuron())

s_a=np.zeros((s_num,a_num),dtype=np.float)
s_c=np.zeros((s_num,c_num),dtype=np.float)

latency=[]
f=open('parameters7.txt',mode='r')
for i in range(s_num):
	for j in range(a_num):
		tmp=f.readline()
		s_a[i,j]=float(tmp)
for i in range(s_num):
	for j in range(c_num):
		tmp=f.readline()
		s_c[i,j]=float(tmp)
for i in range(1000):
	tmp=f.readline()
	latency.append(float(tmp))
f.close()
'''
plt.plot(list(range(300)),latency[0:300])
plt.title('Latency Change')
plt.xlabel('step')
plt.ylabel('latency')
plt.show()
'''
L=[]
U=[]
V=[]
result=[]
err=0
time=[[] for i in range(25)]
for j in range(25):
	state[j].DC=15
	a_flag=[True for i in range(a_num)]
	c_flag=[True for i in range(c_num)]
	for i in range(c_num):
		critic[i].clear()
	for t in range(560):
		signal=np.array([])
		for i in range(s_num):
			signal=np.append(signal,state[i].out([0],[0]))
		for i in range(c_num):
			critic[i].out(signal,s_c[:,i])
			if c_flag[i] and critic[i].fire:
				critic[i].compute_PD()
				c_flag[i]=False
				critic[i].softmax=neuron6.V[t]
	result.append(sum([critic[i].softmax for i in range(c_num)])/c_num)
	err+=(value[j]-result[j])*(value[j]-result[j])
	for i in range(s_num):
		state[i].clear()
	for i in range(a_num):
		action[i].clear()
	for t in range(560):
		signal=np.array([])
		for i in range(s_num):
			signal=np.append(signal,state[i].out([0],[0]))
		for i in range(a_num):
			action[i].out(signal,s_a[:,i])
			if a_flag[i] and action[i].fire:
				action[i].compute_PD()
				a_flag[i]=False
				action[i].softmax=neuron6.Q[t]
				time[j].append(t)
	#print(a_flag)
	qmax=max([action[i].softmax for i in range(a_num)])
	sigma=sum([action[i].softmax for i in range(a_num)])
	if sigma==0:
		sigma=1
	U.append(action[3].softmax/sigma-action[2].softmax/sigma)
	V.append(action[0].softmax/sigma-action[1].softmax/sigma)
	L.append([action[i].softmax/sigma for i in range(4)])
	state[j].DC=0
	for i in range(s_num):
		state[i].clear()
err=err/25
err=np.sqrt(err)
print(err)
#U[0]=0.489251656089623
#V[0]=-0.5107483439103769
X=[]
Y=[]
for i in range(5):
	for j in range(5):
		x_tmp=j
		y_tmp=5-i
		X.append(x_tmp+0.5)
		Y.append(y_tmp-0.5)

for i in range(5):
	print(result[i*5:(i+1)*5])
for i in range(25):
	print(time[i])


plt.xlim(0,5)
plt.ylim(0,5)
plt.scatter([0.5,1.5,2.5,3.5,4.5]*5,[4.5]*5+[3.5]*5+[2.5]*5+[1.5]*5+[0.5]*5,s=[2500]*25,
			c=result,vmin=0,vmax=1,marker='s')
plt.vlines([0,1,2,3,4],0,5,colors='black')
for i in range(5):
	plt.plot([0,1,2,3,4,5],[i]*6,'black')
plt.colorbar()
plt.axis('off')
plt.title('Values Learned')
plt.show()

'''
plt.figure()
ax=plt.gca()
ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=2)
ax.set_xlim([0,5])
ax.set_ylim([0,5])
plt.draw()
plt.vlines([0,1,2,3,4,5],0,5,colors='black')
for i in range(6):
	plt.plot([0,1,2,3,4,5],[i]*6,'black')
plt.axis('off')
plt.title('Probability Learned')
plt.show()
'''