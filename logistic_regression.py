import numpy as np

data = np.loadtxt("data.csv", delimiter=",", skiprows=1) # separates
X = data[:, :3]
y = data[:, 3].astype(int)

w = np.array([0,0,0])

b=0
# Labels: shape (m,)

def sigmoid(n):
    return 1/(1+np.exp(-n))
def calc_gradient(X,y,w,b):
    z=X@w+b
    p=1/(1+np.exp(-z))
    dw=(1/25)*(X.T @ (p-y))
    db=(1/25)*(np.sum(p-y))
    return dw, db
i=0
alpha=0.001
while i<500000:
    dw,db=calc_gradient(X,y,w,b)
    w=w-alpha*dw
    b=b-alpha*db
    i=i+1
z=X@w+b
p=1/(1+np.exp(-z))
print(p)