import numpy as np
import scipy

def buildA(starting_nodes, ending_nodes, n, e):
    As = scipy.sparse.coo_matrix(((np.ones(e),(starting_nodes,np.arange(0,e,1)))), shape=(n,e))
    Ae = scipy.sparse.coo_matrix(((np.ones(e)),(ending_nodes,np.arange(0,e,1))), shape=(n,e))
    Ac = -As + Ae
    A = Ac[:-1,:]
    return A

def reading_net(types):

    index_R = np.asarray(np.where(types=='R')).flatten()
    index_Vs = np.asarray(np.where(types=='V')).flatten()
    index_Is = np.asarray(np.where(types=='I')).flatten()
    index_L = np.asarray(np.where(types=='L')).flatten()
    index_C = np.asarray(np.where(types=='C')).flatten()
    index_D = np.asarray(np.where(types=='D')).flatten()

    return index_R, index_Vs, index_Is, index_L, index_C, index_D

def build_matrices(types, values, e, index_R, index_Vs, index_Is, index_L, index_C, index_D):

    index_R = np.asarray(np.where(types=='R')).flatten()
    index_Vs = np.asarray(np.where(types=='V')).flatten()
    index_Is = np.asarray(np.where(types=='I')).flatten()
    index_L = np.asarray(np.where(types=='L')).flatten()
    index_C = np.asarray(np.where(types=='C')).flatten()
    index_D = np.asarray(np.where(types=='D')).flatten()

    vecR = np.concatenate((index_R, index_Is, index_C, index_D),axis=None)
    R = scipy.sparse.coo_matrix((np.ones(vecR.size), (vecR,vecR)), shape=(e,e))

    vecG = np.concatenate((index_R, index_Vs, index_L, index_D),axis=None)
    valG = np.concatenate((1/values[index_R],np.ones(np.concatenate((index_Vs,index_L),axis=None).size),1/values[index_D]),axis=None)
    G = scipy.sparse.coo_matrix((valG,(vecG,vecG)),shape=(e,e))

    L = scipy.sparse.coo_matrix((values[index_L],(index_L,index_L)),shape=(e,e),dtype=float)
    C = scipy.sparse.coo_matrix((values[index_C],(index_C,index_C)),shape=(e,e),dtype=float)

    vecGen = np.concatenate((index_Vs, index_Is))
    valGen = np.concatenate((values[index_Vs], values[index_Is]))
    s = scipy.sparse.coo_matrix((valGen,(vecGen,np.zeros(vecGen.size))),shape=(e,1))

    return R, G, L, C, s

def tableau_analysis(A, R, G, L, C, n, e):

    r1 = scipy.sparse.hstack((A, scipy.sparse.csr_matrix((n-1,e+n-1),dtype=float)))
    r2 = scipy.sparse.hstack((scipy.sparse.csr_matrix((e,e), dtype=float),-1*scipy.sparse.eye(e),A.T))
    r3 = scipy.sparse.hstack((R,G,scipy.sparse.csr_matrix((e,n-1))))
    M1 = scipy.sparse.vstack((r1,r2,r3)).toarray()

    r1 = scipy.sparse.csr_matrix((n-1,2*e+n-1),dtype=float)
    r2 = scipy.sparse.csr_matrix((e,2*e+n-1),dtype=float)
    r3 = scipy.sparse.hstack((L,C,scipy.sparse.csr_matrix((e,n-1))))
    M2 = scipy.sparse.vstack((r1,r2,r3)).toarray()

    return M1, M2

def modified_tableau_analysis(A, R, G, L, C, n, e):

    A = A.toarray()
    G = G.toarray()
    C = C.toarray()
    R = R.toarray()
    L = L.toarray()

    r1 = np.hstack((A, np.zeros((n-1,n-1), dtype=float)))
    mat = np.matmul(G, A.T)
    r2 = np.hstack((R, mat))
    M1 = np.vstack((r1,r2))

    r1 = np.zeros((n-1,e+n-1), dtype=float)
    mat = np.matmul(C, A.T)
    r2 = np.hstack((L, mat))
    M2 = np.vstack((r1,r2))

    return M1, M2

def nodal_analysis(A, G, C):

    A = A.toarray()
    G = G.toarray()
    C = C.toarray()

    mat = np.matmul(A, G)
    M1 = np.matmul(mat, A.T)

    mat = np.matmul(A, C)
    M2 = np.matmul(mat, A.T)

    return M1, M2

def assemblerhs(vecGen, values, n, e, A, formulation):

    valGen = values[vecGen]
    s = scipy.sparse.coo_matrix((valGen,(vecGen,np.zeros(vecGen.size))),shape=(e,1))
    if formulation == 'Tableau' or formulation == 'tableau':
        rhs = (scipy.sparse.vstack((scipy.sparse.csr_matrix((n-1+e,1)),s))).toarray()

    if formulation == 'Modified Tableau' or formulation == 'Modified tableau' or formulation == 'modified tableau' or formulation == 'MT':
        rhs = (scipy.sparse.vstack((scipy.sparse.csr_matrix((n-1,1)),s))).toarray()
        
    elif formulation == 'Na' or formulation == 'nodal analysis' or formulation == 'NA':
        rhs = np.matmul(A.toarray(), s.toarray())
    return rhs

def assembleK(theta,dt,M1,M2):
    
    K1 = (theta*M1 + 1/dt*M2)
    K2 = (-(1-theta)*M1 + 1/dt*M2)

    return K1, K2

def theta_method(M1, M2, theta, x0, dt, t, s_t, values, vecGen, e, n, A, formulation):
    # first iteration
    K1, K2 = assembleK(theta, dt/1e10, M1, M2)
    values[vecGen] = s_t[0]
    rhs_old = assemblerhs(vecGen, values, n, e, A, formulation)
    values[vecGen] = s_t[1]
    rhs_new = assemblerhs(vecGen, values, n, e, A, formulation)
    q = (theta*rhs_new + (1-theta)*rhs_old).flatten()
    b = np.matmul(K2,x0) + q
    x = np.zeros(shape=(M1.shape[0], t.size))
    x[:,0] = x0
    x[:,1] = np.linalg.solve(K1,b)
    # following iterations
    K1,K2 = assembleK(theta,dt,M1,M2)
    rhs_old = rhs_new
    for k in range(1, t.size-1):
        print(k)
        values[vecGen] = s_t[k+1]
        rhs_new = assemblerhs(vecGen, values, n, e, A, formulation)
        q = (theta*rhs_new + (1-theta)*rhs_old).flatten()
        b = np.matmul(K2, x[:,k]) + q
        x[:,k+1] = np.linalg.solve(K1,b)
        rhs_old = rhs_new
    return x

def RK_solver_NA(M1, M2, A, s, i_t, x0, vecGen, t_span, t_eval=None):
        
    A = A.toarray()
    
    mat1 = -np.matmul(np.linalg.inv(M2), M1)
    mat2 = np.matmul(np.linalg.inv(M2), A)
    
    s = s.toarray()
    
    s1 = s[:vecGen[0]]#.flatten()
    s2 = s[vecGen[0]+1:]#.flatten()
    
    s_t = lambda t: np.vstack([s1, i_t(t), s2])

    F = lambda x,t: np.matmul(mat2, s_t(t)) #np.matmul(mat1,x) +

    sol = scipy.integrate.solve_ivp(F, t_span, x0, t_eval=t_eval)

    return sol.t, sol.y

