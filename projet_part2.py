import numpy as np

#-----------------------------------------------------------------
#Partie 2
with open('2.2.2_sequences.txt') as f:
    sequences = f.readlines()
    
sequences = [sequences[i] for i in range(1, 10, 2)]#len(sequences), 2)]
sequences = [seq.rstrip('\r\n') for seq in sequences]

K = 4
W = 10
N = len(sequences)

#remplissage de n_i,j^{-k} : nombre de nucléotides i à la position j des motifs (sauf le motif k)
#remplissage de n~_i,0^{-k} : nombre de nucléotides i dans la région hors des motifs (avec motif en k compris par contre)
def nEtnTilde_k(Amoinsk, S, k):
    n_k = np.zeros((K, W))
    nTilde_k = np.zeros(K)
    for l in range(N): #parcours des séquences
        for j in range(len(S[l])): #parcours au sein d'une sequence
            seq = S[l][j]
            if l != k and (j >= Amoinsk[l] and j < Amoinsk[l] + W): #remplissage de n
                idx = int(j - Amoinsk[l])
                if seq == 'A':
                    n_k[0][idx] += 1
                elif seq == 'C':
                    n_k[1][idx] += 1
                elif seq == 'G':
                    n_k[2][idx] += 1
                elif seq == 'T':
                    n_k[3][idx] += 1
            else: #remplissage de n~
                if seq == 'A':
                    nTilde_k[0] += 1
                elif seq == 'C':
                    nTilde_k[1] += 1
                elif seq == 'G':
                    nTilde_k[2] += 1
                elif seq == 'T':
                    nTilde_k[3] += 1
    return n_k, nTilde_k

#remplissage de n_i,j : nombre de nucléotides i à la position j des motifs
#remplissage de n~_i,0 : nombre de nucléotides i dans la région hors des motifs
def nEtnTilde(A, S):
    n = np.zeros((K, W), dtype=int)
    nTilde = np.zeros(K, dtype=int)
    for l in range(N): #parcours des séquences
        for j in range(len(S[l])): #parcours au sein d'une sequence
            seq = S[l][j]
            if j >= A[l] and j < A[l] + W: #remplissage de n
                idx = int(j - A[l])
                if seq == 'A':
                    n[0][idx] += 1
                elif seq == 'C':
                    n[1][idx] += 1
                elif seq == 'G':
                    n[2][idx] += 1
                elif seq == 'T':
                    n[3][idx] += 1
            else: #remplissage de n~
                if seq == 'A':
                    nTilde[0] += 1
                elif seq == 'C':
                    nTilde[1] += 1
                elif seq == 'G':
                    nTilde[2] += 1
                elif seq == 'T':
                    nTilde[3] += 1
    return n, nTilde

#remplissage de z_i,j^k : 1 si nucléotide en i à la position A_k + j de S_k, 0 sinon
def z_i_j_k(i, j, A_k, S_k):
    seq = S_k[int(A_k + j)] # - 1
    if (i == 0 and seq == 'A') or (i == 1 and seq == 'C') or (i == 2 and seq == 'G') or (i == 3 and seq == 'T'):
        return 1
    return 0

beta = 1
alpha = 1
def Etheta(i, j, n_k):
    sumB = 0
    for l in range(K - 1):
        sumB += beta

    return (n_k[i][j] + beta)/(N - 1 + sumB)

def Ephi(i, nTilde_k, lengthS):
    sumA = 0
    for l in range(K):
        sumA += alpha
    
    return (nTilde_k[i] + alpha)/(lengthS - (N-1) * W + sumA)

def thetaFinal(i, j, n):
    sumB = 0
    for l in range(K - 1):
        sumB += beta

    return (n[i][j] + beta)/(N + sumB)

def phiFinal(i, nTilde, lengthS):
    sumA = 0
    for l in range(K):
        sumA += alpha
    
    return (nTilde[i] + alpha)/(lengthS - N * W + sumA)

lengthS = 0
for i in range(len(sequences)):
    lengthS += len(sequences[i])

def probaAk(Amoinsk, S, k):
    proba = np.ones(len(S[k]) - W + 1) #init à 1 car *
    n_k, nTilde_k = nEtnTilde_k(Amoinsk, S, k)
    for l in range(len(proba)): # les differents A_k possibles
        for i in range(K):
            for j in range(W):
                Ak = l
                proba[l] *= (Etheta(i, j, n_k)/Ephi(i, nTilde_k, lengthS))**(z_i_j_k(i, j, Ak, S[k]))
    return proba/sum(proba)

A = np.zeros(N) 
for i in range(N):
    A[i] = int(np.random.randint(len(sequences[i]) - W))

Amoinsk = A #on passera l'élément k

proba = probaAk(Amoinsk, sequences, 1)

def updateY(i, y_A, S):
    proba = probaAk(y_A, S, i)
    aleatoire = np.random.rand()
    tmp = proba[0]
    for k in range(len(proba)):
        if tmp < aleatoire:
            if k + 1 != len(proba):
                tmp += proba[k+1]
        else: #tmp >= aleatoire
            return k
    #si du aux erreurs de decimales on est parvenu au bout sans return
    return len(proba) - 1

def rapportFactoriel(numerateur, denominateur):
    if numerateur == denominateur:
        return 1
    elif numerateur > denominateur:
        tmp = 1
        for i in range(denominateur + 1, numerateur + 1, 1):
            tmp *= i
        return tmp
    else:
        tmp = 1
        for i in range(numerateur + 1, denominateur + 1, 1):
            tmp *= i
        return 1/tmp

def rapport_P(A, S, Ashift):
    rapport = 1
    n, nTilde = nEtnTilde(A, S)
    nshift, nTildeshift = nEtnTilde(Ashift, S)
    for i in range(K):
        rapport *= rapportFactoriel(nTildeshift[i] + alpha, nTilde[i] + alpha)
        for j in range(W):
            rapport *= rapportFactoriel(nshift[i][j] + beta, n[i][j] + beta)
            if rapport > 10**(20):
                break
            if rapport < 10**(-20):
                break
        if rapport > 10**(20):
            break
        if rapport < 10**(-20):
            break
    return rapport

#----------------------
#critere de performance
def scoreC(theta):
    tmp = 0
    for i in range(K):
        for j in range(W):
            tmp += theta[i][j] * np.log2(theta[i][j])
    return 2 - tmp/W

def scoreKL(theta, phi):
    tmp = 0
    for i in range(K):
        for j in range(W):
            tmp += theta[i][j] * np.log(theta[i][j]/phi[i])
    return tmp/W

def phi_theta_n_ntilde(S, A):
    n, nTilde = nEtnTilde(A, S)
    phi = np.zeros(K)
    theta = np.zeros((K, W))
    for i in range(K):
        phi[i] = phiFinal(i, nTilde, lengthS)
        for j in range(W):
            theta[i][j] = thetaFinal(i, j, n)
    return phi, theta, n, nTilde

T = 1000
M = 500
shift = np.array([-1, 1]) #on testera ces valeurs de shift
def algoGibbs(S, A):
    y_A = np.zeros((T+1, N), dtype=int)
    score = np.zeros((T+1))
    y_Ashift = np.zeros(N, dtype=int)
    y_Aselected = np.zeros(N, dtype=int)
    for i in range(N):
        y_A[0][i] = A[i]
    phi, theta, _, _ = phi_theta_n_ntilde(S, y_A[0])
    score[0] = scoreKL(theta, phi)

    i = 0
    for t in range(T):
        y_A[t+1] = y_A[t]
        # here a step for the shift can be inserted
        i = (i + 1) % N # balayage systématique
        #i = np.random.randint(N) # balayage non systématique
        y_A[t+1][i] = updateY(i, y_A[t+1], S)
        phi, theta , _, _ = phi_theta_n_ntilde(S, y_A[t+1])
        score[t+1] = scoreKL(theta, phi)
    idx_choix = np.argwhere(score[:] == max(score))
    idx_choix = idx_choix[0][0]
    phi, theta, n, nTilde = phi_theta_n_ntilde(S, y_A[idx_choix])
    return y_A[idx_choix], phi, theta, n, nTilde, max(score)

y_A, phi, theta, n, nTilde, scorey_A = algoGibbs(sequences, A)
print("y_A[max_score] =", y_A, "\n")
print("phi =", phi, "\n")
print("theta =", theta, "\n")
print("n =", n, "\n")
print("nTilde =", nTilde, "\n")
print("max_score_y_A =", scorey_A, "\n")

S = sequences
y_Ashift = np.zeros(N, dtype=int)
#insert a step for the shift
for s in shift:
    for q in range(N):
        if y_A[q] + s < 0 or y_A[q] + s > len(S[q]) - W + 1:
            NotWorked = True
            break
        else:
            y_Ashift[q] = y_A[q] + s
    phi_shift, theta_shift, n_shift, nTilde_shift = phi_theta_n_ntilde(S, y_A)
    score = scoreKL(theta, phi)
    if score > scorey_A:
        scorey_A = score
        phi = phi_shift
        theta = theta_shift
        n = n_shift
        nTilde = nTilde_shift
        y_A = y_Ashift

print("y_A[max_score] =", y_A, "\n")
print("phi =", phi, "\n")
print("theta =", theta, "\n")
print("n =", n, "\n")
print("nTilde =", nTilde, "\n")
print("max_score_y_A =", scorey_A, "\n")

for i in range(N):
    print(str(y_A[i])+","+str(y_A[i] + W - 1))

def scoreP_S_A(theta, phi, n, nTilde):
    tmp = 0
    for i in range(K):
        tmp += nTilde[i]*np.log(phi[i])
        for j in range(W):
            tmp += n[i][j]*np.log(theta[i][j])
    return tmp

print(scoreP_S_A(theta, phi, n, nTilde))