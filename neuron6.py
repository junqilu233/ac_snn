import numpy as np

tao_m=0.15      #tao_m=150ms
tao_s=tao_m/4    #tao_s=150/4ms
time=tao_m*tao_s/(tao_m-tao_s)*np.log(4)
kappa=np.exp(-time/tao_m)-np.exp(-time/tao_s)
theta=2
K=np.zeros(3001)
K_up=np.zeros(3001)
#time t
for i in range(3000):
	K[i]=(np.exp(-i/1000/tao_m)-np.exp(-i/1000/tao_s))/kappa
	K_up[i]=(np.exp(-i/1000/tao_s)/tao_s-np.exp(-i/1000/tao_m)/tao_m)
K[3000]=0
K_up[3000]=0

Q=np.zeros(3001)
for i in range(3000):
	Q[i]=np.exp(-i/1000)
Q[3000]=0
V=Q

#recover tao=50ms
R=np.zeros(3001)
for i in range(3000):
	R[i]=np.exp(-i/50)
R[3000]=0

class neuron:
	def __init__(self):
		self.lft=3000                 #last_firing_time
		self.spike=np.zeros((0,),dtype=np.int)      #firing_spikes_time
		self.t_rec=[]               #spikes_received_time
		self.w_rec=[]               #spikes_received_time's_weight
		self.volt=0                 #membrane_volt
		self.v0=0                   #initial_membrane_volt
		self.PD=[]                  #partial derrivative
		self.softmax=0             #e^(-t)
		self.DC=0                   #receive_a_DC_stimulus
		self.fire=False             #fire_a_spike_or_not
	
	def clear(self):
		self.lft=3000
		self.spike=np.zeros((0,),dtype=np.int)      #firing_spikes_time
		self.t_rec=[]               #spikes_received_time
		self.w_rec=[]               #spikes_received_time's_weight
		self.volt=0                 #membrane_volt
		self.v0=0                   #initial_membrane_volt
		self.PD=[]                  #partial derrivative
		self.softmax=0             #e^(-t)
		self.fire=False
		
	def compute_PD(self):
		self.PD=[]
		for i in range(len(self.t_rec)):
			inf=0
			for j in range(len(self.t_rec[i])):
				inf=inf+K_up[self.t_rec[i][j]]
			if inf==0:
				partial=0
			else:
				partial=kappa*theta/inf
			self.PD.append(partial)
		
	def out(self,s,w):
		#time pass
		#print(type(self.spike),self.spike)
		self.spike=self.spike+1
		self.lft=self.lft+1
		if self.lft>3000:
			self.lft=3000
		for i in range(len(self.t_rec)):
			self.t_rec[i]=self.t_rec[i]+1
		#print(self.t_rec)					
		
		#if fire a spike in the last moment, then clear all the old influences
		if self.fire:
			self.v0=-5
			self.volt=self.v0
			self.t_rec=[]
			self.w_rec=[]
			for i in range(len(s)):
				if s[i]==1:
					self.t_rec.append(np.zeros((1),dtype=np.int))
					self.w_rec.append(w[i]*np.ones((1),dtype=np.float))
				else:
					self.t_rec.append(np.zeros((0,),dtype=np.int))
					self.w_rec.append(np.zeros((0,),dtype=np.float))
			self.fire=False
			return 0
		
		#recover from V_initial
		self.volt=self.v0*R[self.lft]
		#membrene volt reaches a constant DC stimulus
		if self.DC!=0:
			self.volt=self.volt+self.DC*(1-R[self.lft])
		#calculate membrene volt and update spikes received time
		if len(self.t_rec)==0:
			self.t_rec=[]
			self.w_rec=[]
			for i in range(len(s)):
				self.t_rec.append(np.zeros((0,),dtype=np.int))
				self.w_rec.append(np.zeros((0,),dtype=np.float))
			#print(self.t_rec)
		for i in range(len(self.t_rec)):
			for j in range(len(self.t_rec[i])):
				#print(i,self.t_rec[i])
				self.volt=self.volt+self.w_rec[i][j]*K[self.t_rec[i][j]]
			if s[i]==1:
				self.t_rec[i]=np.append(self.t_rec[i],0)
				self.w_rec[i]=np.append(self.w_rec[i],w[i])
		#if membrene volt crosses threshald then fire a spike
		if self.volt>theta:
			self.volt=10
			self.fire=True
			self.lft=0
			self.spike=np.append(self.spike,0)
			#print(type(self.spike),self.spike)
			return 1
		return 0

