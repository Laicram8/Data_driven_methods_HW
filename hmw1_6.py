import numpy as np
from numpy import linalg 
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.optimize import minimize

y = np.linspace(-2,2,401) # spatial coordinate
Ny = np.size(y)

amp1 = 1
y01 = 0.5
sigmay1 = 0.6

amp2 = 1.2
y02 = -0.5
sigmay2 = 0.3

dt = 0.1
Nt = 101
tend = dt*(Nt-1)
t = np.linspace(0,tend,Nt) # time

omega1 = 1.3
omega2 = 4.1

v1 = amp1*np.exp(-((y-y01)**2)/(2*sigmay1**2))
v2 = amp2*np.exp(-((y-y02)**2)/(2*sigmay2**2))

X = np.zeros([Ny,Nt],dtype=complex)
for tt in range(Nt):
    X[:,tt] = v1*np.exp(1j*omega1*t[tt])+v2*np.exp(1j*omega2*t[tt]) 

############################################################################################
# -------------------- 2D CONTOURS
############################################################################################

# ------ Real part
T, Y = np.meshgrid(t, y)
fig, r = plt.subplots(figsize=(6,5))
plot_data=r.contourf(T,Y,np.real(X),200,cmap='bwr')
fig.colorbar(plot_data, ax=r)
r.set_title('Data (real part)')
r.set_ylabel('Space (y)')
r.set_xlabel('Time (t)')
plt.show()

############################################################################################
# -------------------- SVD
############################################################################################
"""
U, S, VT = np.linalg.svd(X,full_matrices=False)

# -------------------- Singular Vectors

plt.plot(y,np.real(U[:,0:2]))
plt.title('First two singular vectors (real part)')
plt.ylabel('Singular vector')
plt.xlabel('Space (y)')
plt.legend(['1st','2nd'])
plt.show()

# -------------------- Singular values 

plt.semilogy(S/sum(S),'.')
plt.title('Singular Values')
plt.xlabel('Rank (r)')
plt.show()

"""
############################################################################################
# -------------------- DMD
############################################################################################

# -------------------- rank

r=2

# -------------------- We define matrices

X1=X[:,:-1]
X2=X[:,1:]

# -------------------- SVD - Reduce rank 

U, S, VT = np.linalg.svd(X1,full_matrices=0)
   
Ur = U[:,:r]
Sr = np.diag(S[:r])
VTr = VT[:r,:]

# -------------------- Build Atilde 

Atilde = np.linalg.solve(Sr.T,(np.conjugate(Ur.T) @ X2 @ np.conjugate(VTr.T) ).T).T

Lambda, W = np.linalg.eig(Atilde) 

# -------------------- Build DMD modes

Phi = X2 @ np.linalg.solve(Sr.T,np.conjugate(VTr)).T @ W     

# -------------------- Build amplitude

alpha1 = Sr @ np.conjugate(VTr[:,0])
b = np.linalg.solve(W @ np.diag(Lambda),alpha1)

# -------------------- Reconstruct X -> X_dmd


dt=t[1]-t[0]
omega=np.log(Lambda)/dt

time_dynamics=np.zeros(shape=(r,len(t)))
for iter in range(len(t)):
    time_dynamics[:,iter]=b * np.exp(omega * t[iter])
X_dmd=Phi@time_dynamics

T, Y = np.meshgrid(t, y)
fig, i = plt.subplots(figsize=(6,5))
plot_dataa=i.contourf(np.real(X_dmd),200,cmap='bwr')
fig.colorbar(plot_dataa, ax=i)
i.set_title('Data (real part)')
i.set_ylabel('Space (y)')
i.set_xlabel('Time (t)')
plt.show()



