import neuron6
import numpy as np
import random
import matplotlib.pyplot as plt

alpha=0.01
beta=0.01
gamma=0.9


state=[]
action=[]
critic=[]

s_num=25
a_num=4
c_num=10
'''
Reward=[0.2,0.3,0.2,0.1,0,
		0.3,0.4,0.3,0.2,0.1,
		0.4,0.5,0.4,0.3,0.2,
		0.5,0.7,0.5,0.4,0.3,
		0.7,1,0.7,0.5,0.4]
'''
Reward=[0,0,0,0,0,
		0,0,0,0,0,
		0,0,1,0,0,
		0,0,0,0,0,
		0,0,0,0,0]
'''
Reward=[0.3,0.3,0.3,0,0,
		0.3,0.3,0.3,0.3,0,
		0.3,0.6,0.3,0.3,0.3,
		0.6,0.6,0.6,0.3,0.3,
		0.6,1,0.6,0.6,0.3]
'''
for i in range(s_num):
	state.append(neuron6.neuron())
for i in range(a_num):
	action.append(neuron6.neuron())
	action[-1].PD=[0]*s_num
for i in range(c_num):
	critic.append(neuron6.neuron())
	
s_c=np.zeros((s_num,c_num),dtype=np.float)
s_a=np.zeros((s_num,a_num),dtype=np.float)

f=open('parameters7.txt',mode='r')
for i in range(s_num):
	for j in range(a_num):
		tmp=f.readline()
		s_a[i,j]=float(tmp)
for i in range(s_num):
	for j in range(c_num):
		tmp=f.readline()
		s_c[i,j]=float(tmp)
f.close()

ll=[]
try:
	for size in range(1000):
		batch=list(range(s_num))
		del(batch[12])
		dV=[[0]*c_num for i in range(s_num)]
		V=[[0]*c_num for i in range(s_num)]
		direction=[-5,5,-1,1]
		next_state=[12]*s_num
		dlog=[[0]*a_num for i in range(s_num)]
		for s in batch:
			c_flag=[True for i in range(c_num)]
			a_flag=[True for i in range(a_num)]
			state[s].DC=15
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
				for i in range(a_num):
					action[i].out(signal,s_a[:,i])
					if a_flag[i] and action[i].fire:
						action[i].compute_PD()
						a_flag[i]=False
						action[i].softmax=neuron6.Q[t]
			for i in range(c_num):
				V[s][i]=critic[i].softmax
				if len(critic[i].PD)==0:
					dV[s][i]=2.5
				else:
					dV[s][i]=V[s][i]*critic[i].PD[s]/s_c[s,i]/s_c[s,i]
			qmax=max([action[i].softmax for i in range(a_num)])
			sigma=sum([action[i].softmax for i in range(a_num)])
			if sigma==0:
				sigma=1
			choose=[]
			if qmax==0:
				for i in range(a_num):
					if s_a[s][i]>0:
						choose.append(i)
				act=random.choice(choose)
			else:
				tes=np.random.rand()
				for i in range(a_num):
					tes=tes-action[i].softmax/sigma
					if tes<0:
						act=i
						break
			next_state[s]=s+direction[act]
			for i in range(a_num):
				if s_a[s,i]==0:
					dlog[s][i]=0
				elif len(action[i].PD)==0:
					dlog[s][i]=(i==act)
				elif i==act:
					dlog[s][i]=(1-action[i].softmax/sigma)*action[i].PD[s]/s_a[s,i]/s_a[s,i]
				else:
					dlog[s][i]=(-action[i].softmax/sigma)*action[i].PD[s]/s_a[s,i]/s_a[s,i]
			state[s].DC=0
			for i in range(s_num):
				state[i].clear()
			for i in range(c_num):
				critic[i].clear()
			for i in range(a_num):
				action[i].clear()
		for i in range(5):
			print([sum([V[j][k] for k in range(c_num)])/c_num for j in range(i*5,(i+1)*5)])
		for i in range(5):
			print(next_state[i*5:(i+1)*5])
		print('\n')
		delta=[[0]*c_num for i in range(s_num)]
		dw=[[0]*c_num for i in range(s_num)]
		dtheta=[[0]*a_num for i in range(s_num)]
		for s in batch:
			for i in range(c_num):
				delta[s][i]=Reward[next_state[s]]+gamma*V[next_state[s]][i]-V[s][i]
				dw[s][i]+=alpha*delta[s][i]*dV[s][i]
				#print(s,i,dw[s][i])
			for i in range(a_num):
				delta0=sum(delta[s])/c_num
				if s_a[s,i]>0:
					dtheta[s][i]=beta*delta0*dlog[s][i]
		for s in batch:
			for i in range(c_num):
				s_c[s,i]+=dw[s][i]
				if s_c[s,i]<0:
					s_c[s,i]=0
				elif s_c[s,i]>3:
					s_c[s,i]=3
			for i in range(a_num):
				if s_a[s,i]>0:
					s_a[s,i]+=dtheta[s][i]
					if s_a[s,i]<0.1:
						s_a[s,i]=0.1
					elif s_a[s,i]>0.35:
						s_a[s,i]=0.35
		s=4
		latency=0
		while not s==12:
			for i in range(s_num):
				state[i].clear()
			for i in range(a_num):
				action[i].clear()
			a_flag=[True for i in range(a_num)]
			state[s].DC=15
			for t in range(560):
				signal=np.array([])
				for i in range(s_num):
					signal=np.append(signal,state[i].out([0],[0]))
				for i in range(a_num):
					action[i].out(signal,s_a[:,i])
					if a_flag[i] and action[i].fire:
						a_flag[i]=False
						action[i].softmax=neuron6.Q[t]
				if sum(a_flag)==0:
					break
			state[s].DC=0
			qmax=max([action[i].softmax for i in range(a_num)])
			sigma=sum([action[i].softmax for i in range(a_num)])
			if sigma==0:
				sigma=1
			choose=[]
			if qmax==0:
				for i in range(a_num):
					if s_a[s][i]>0:
						choose.append(i)
				act=random.choice(choose)
			else:
				tes=np.random.rand()
				tmp=tes
				for i in range(a_num):
					tmp=tmp-action[i].softmax/sigma
					if tmp<0:
						act=i
						break
			direction=[-5,5,-1,1]
			s=s+direction[act]
			latency=latency+1
		for i in range(s_num):
			state[i].clear()
		for i in range(a_num):
			action[i].clear()
		ll.append(latency-4)
		print(ll[-1])
		print('\n')
except KeyboardInterrupt as e:
	pass

f=open('parameters7.txt',mode='w')
for i in range(s_num):
	for j in range(a_num):
		f.writelines(f'{s_a[i,j]}\n')
for i in range(s_num):
	for j in range(c_num):
		f.writelines(f'{s_c[i,j]}\n')
for i in range(len(ll)):
	f.writelines(f'{ll[i]}\n')
f.close()
'''
f=open('latency.txt',mode='a')
for i in range(len(ll)):
	f.writelines(f'{ll[i]}\n')
f.close()
plt.plot(range(len(ll)),ll)
plt.show()
'''