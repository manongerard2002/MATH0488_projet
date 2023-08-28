import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=16)         # controls default text sizes
plt.rc('axes', titlesize=16)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)   # fontsize of the tick labels
plt.rc('ytick', labelsize=16)   # fontsize of the tick labels
plt.rc('legend', fontsize=12)   # legend fontsize
plt.rc('figure', titlesize=32)  # fontsize of the figure title
plt.rc('lines', linewidth=3)    # linewidth of the plot

seq1 = 'agcgcctgaccagtagagagacttagcgcaacacgctacgagaacctaaccagccactccagctcatacattacgcccccgctacgaaccaaccaaccgctacacacgctaccagacctcgggctacgcgcagcatcacagccaagcggcacaacggctatctgccgcacatacccgcgcgacctagataccgccacccactagctataccgtactacacacgtagcctaccagaactacactaccagacgtacaccgcgtaccccgcgatactacacagcctatataccgcaccggttaactatacacacacactcgcatagaggcatagtactatactacagcgccctactacccttagagaccctacctacatagtggctaagacctataggcgcgcactttgcagctgcgctacaagcagacgcatgcttactaccgacagatagatttcccgatgcgtttacctgcctatactcgacggcacagactcagcggcacacagtaccacgacgttagcaagagagtatgcccccaacttacccagcgagagggacgactacctagacagtaacgagacatacgcatagcatcagcgacaccacgcgcacctacatagcttcattagaataccgcacgcctaacgcggagatgaggactagctccaaacactactacagagtgatacttagagcttagacccaccgagctcgctatgacatcacgctaccgactggccgcgagccgcactagacgagagataccgagagccgccgcccctgagcacgctatcgaaccgtagcatacgacgctattccatagcaactagagaccagctacgaccagcgccacacgactactacacgctgaagtcccctaccgcgctaccgactactgctcctacgagacacgatacatagaacagcacgagctacgactagcgcgcgggctaccgaccggcgcctctacatatactcgagcaacgcagcggactactactagaacagtacgctagcctaacccgccagcaactacgccctagatacactactacgtacgctagctagagtactacgacagaccgcggactaaccactgtaataacccacccgacctactactacgcgcgaggagaccgactacactacccagctaaccaacccgtagctaacgaggaccactttaccacgatacaccaccagcgctactacgatactagcagctgtcgctagctaaactacaccggctaccctactacgcagtataccaacacaacaagacaccctgtactggagcagacttaccccaagctacgccttacgtacagagcgcgccgcctaagcacactgccgcgcgggctagaccataagcactaccacacactacgctctagctagagtaataccaagcagctacggcaccctactccacgactccgaagacgcgcgccgagtaccgcccgactatacacgcacactactacgcatagagccaataggccgcgcacactaccgtgcgcgacagcacacgtgcaccgcgatgctactgcgggcccgcgcttaagagctgcagccgcgctatatcacttagggattagccactgcgggcagacagagacagcacagcaagtactacgacgagcacgcgtttaacctctagcacacgtaccaagctaggagatactaccggctagccattaccctacagactacgcgcctcagaatagcgcacaacgtacgggtagccccatacgctatattagccctatagcgcaggaccagattaccgcgcgcccgctacatagccacgacgccaagcagcgcggctacgctatagcacgcaccgtactactgctagactagtactacgacgggaacagtacagctaactactgacgcgacgagcacgggagcctacccgccagccatcataaactccactatacgcgacccgagcgaggacgctttatacgagactccacttgccacagcgccacgaga'
seq2 = 'gtaactaccgcccaagacatacaatggcatcgatcgtgtcccgcaccaaacaaaccggaagaaattctttgcaggcctcgacctcatagctcaagccatccgaggaactgcatcgaaaacgacccatccaggcctcactcaattctccccggaggcccccacacaacggtacggagagcccgcacgaaaaacgctagcctcccgctaacgctgcgcccgaagcctatgaggaaaaagtaaggttcagcctccgacgccatgccaatgttacatcacacgcgtaacagacagtctctaaaagaacggtaccccgaaagtttcagcaacgactccgcacacggcggcaacccaccaccaatttaccctgcctggaggacacgctccgaactcaacacaaaccatctgaccataatagctaggaaaattgcacgctgtggccctgagaagccaacagatgagggcacaatccataaaaacggcacacataaacacaatcggacacatcacagtaaggactccaggtgctgcaagcccctcagccccctccccgattgacgcgatctcccccacgacgtcgccaactaggagatgtacgacgcgaactacctcccagtccaaagtaaactgtccaggcccgttcaagctccataatcatacagcaactacaagcaagaatggaccgaagcgccttgctcagccccagccctccttaaaggcagtgtcgctccatccgccgtaaaactctgactcgacctaaatcacgcactatactatcaagtaaaaaacagctcctcgaagcgcgccacgcgccgtaggaacggcacacacccgccaaaggaaaccctcttcagcgtcgaagcacgacgtaaatgatcgccgtactaaggagggaatgccctcaacctcatacttctaccgaacgggaatccaaggtgacaccgaaccgggcgcttactaagatatcgacgccaccaacccacacagcagggacgaggccggacagaaagatagtcgtatgtctcaccccatccgaatgacacgcggcacagacgtgagaataccggccccctctcactcactaaacgagcaggaagaagctacccacccactggcccaagcccgaagcgcgtcagcagaaaatgatataaaacccagcgctgccactctccacgtacgcaaccagatgcccagaaaactacaaacgtgtatcccggcagccctccatgtcctaccccctacttcgaacctacccaccacatcgaagcgccaccacaaaagagatttatgccacgccggcccgcggagcgctataaccgccctgttgaccgcctaaaaaaaagacttccggcccgaaacctaacgaatcgccaagcgcaaccgatcgccgtccttaggcaatccacgcgcatgaacaagcgcatcatttcgcacaacctcgcaccaatctcccgaagattaaatgcggattctcaccttctagcaagcgactctggagataaacgtcgaagaaaacaacgcccacgaacctgccaatagttggaacgcccgcggagtgcgccgatcgcgtatgaacccgggaggccaaactcaatgtacccccaacccaagtgaaccgctggctcaaaaccactaaaagtacgtgcgttttctcctctaccaaacacttaactatcagacgccccggtataccacctcacgcatgaccacccatctttccaccagacgctagaatgacacgcaaacaacgaaatagacatcccggccaaccagacacggctaaactagagctaccccataagacgtccaaatccacacgctagacttctcgctaatagccacaaccatcctacggaaagactgaatccatcttagacgcgaacgtcttatcaagctaaaggctctaagttaggcacgctaaccagcgtaggaacgacccatagttcccacagctacaaaacgatacgcgtttccatacctgccaggtcctgctgcaacatcac'
seq3 = 'cgccggctacgagactagcacgttagcccacggcgcatactcgccagctaaggccctactacgctacgccgacgacactacgctactactactcctacgaacacagcagcaagaccacacactacgacactcctagagctagcacgcatacgctgcagcagccgactctagcccgctacgcagcctacaacaacactaacaagaacaacgctactactacgcccgcgcacctaaccaaagatcctccacccacgagacactacactgacgagatacacagcgagctacacgacgatcagcaccttaccttcgagcatagcagccacgtagccgccactagcccccgagctaatactacacgtacgatagtaccgctacccgcttagcgacgagctctcagcgaggtaccactagccaggactacataaccagttgagacgctaacagtagcctagatagagcgcgccgccgctacagtgctactagaaacagtacgaagcgtagcgaccctacacgcatacacgcgcgtaactaggagctacctacctacttattacggcctagcggcctctagacctacactacacactaccgcaccctaccccacgagcacccggcatgcgtacacgctaatagaactagcgcgtaggatatctccagcgtacacgctacaacctaactaccctgactggcacagcacatggccacttagttgagcgtactaaccactactagcacatgccactcactgctaatcccagagagaagtgccgagtactgactataatcgcttaacaagaagttacccgctactacctgtaccactcactagagctatagctaattagctactagaaccagaggcccgctagccccctcgagaacccacggattactaacccacgcacgaccacgacactataggcgccagaccgcacagcgcgaccacgcgtacactagctaccacgcgcccgaagaccaccggcacaccaccgctagctagacattctcaagctactacgaacaccatagccataggctaacgaatgaccgagcgttatactactactctagacccgtaccgataacaacactagcacacgcgcgacagctacagcccaccgcgccagatctagcgctgcttacacgaccgcgcacggacgtacgaccgactagcttactgagccgcctagctactactacgccctagtacacgcataccgactgcagcgtaccactagagtaccgcacgcaagcagacacgctacgctacggccttacaccactcacgagtaagctgtacaccacgcctacgtgcctacaaggcgccgaccactaccagagcaccttatcataccttaggacctagctacgagtaacatgatagcgcctaacgatatacgagcctagatccagacccacaggcagactactacggcgatacactacacgctagagacacactaccagactaccatactacacgaccgcacgagtagcgctacgaacgcccacgcgagcccgctacagccttaacgcccccgtagcactttaactacgctacgccgcagatatacctacacggcgcgcgcgctagcgcacacgctaatagagaggcactagcctgtactatactacggagccaccacaccgacctacaacccactacgctacgcacaaggtacgccacccgtacgtacacagacgcggcggactagcagcagcgcgcagtacccacacgctacgccacacctactatggcactaaacgcctccgacggctactactaccgcacagaccactcttgctatatatagtagctaaagcactgcttacctttagtaccctaagccagctaccgcgccgtacggccgcgcctatagcccaacgccagatacgcctatgcagtgccggactacctagctacgcttgctagccacctggtcaccactatcccaccgcttaggacacacgtagcactgcgataggtaacgcgcgcc'

#Point 1.1 question 1 :
def calculNbPaire(seq, i, indice, nombrePaire):
    if seq[i+1] == 'a':
        nombrePaire[indice][0] += 1
    if seq[i+1] == 'c':
        nombrePaire[indice][1] += 1
    if seq[i+1] == 'g':
        nombrePaire[indice][2] += 1
    if seq[i+1] == 't':
        nombrePaire[indice][3] += 1

def calculPaire(seq):
    nombrePaire = np.zeros((4, 4))
    nombreBase = np.zeros(4)
    for i in range(len(seq) - 1):
        if seq[i] == 'a':
            calculNbPaire(seq, i, 0, nombrePaire)
            nombreBase[0] += 1
        if seq[i] == 'c':
            calculNbPaire(seq, i, 1, nombrePaire)
            nombreBase[1] += 1
        if seq[i] == 'g':
            calculNbPaire(seq, i, 2, nombrePaire)
            nombreBase[2] += 1
        if seq[i] == 't':
            calculNbPaire(seq, i, 3, nombrePaire)
            nombreBase[3] += 1
    return nombrePaire, nombreBase

def calculQ(seq):
    Q = np.zeros((4, 4))
    nombrePaire, nombreBase = calculPaire(seq)
    for i in range(4):
        for j in range(4):
            if nombreBase[i] != 0:
                Q[i][j] = nombrePaire[i][j]/nombreBase[i]
    return Q


Q = calculQ(seq1)
print('Q = ', Q, "\n")


#Point 1.1 question 2 :
def Q_multistep(Q, t):
    return np.linalg.matrix_power(Q, t)


def P_Xt_Unif(Q, t):
    P_Xt = np.zeros((4, t+1))
    P_Xt[:, 0] = 1/4 #pi_0
    for i in range(1, t+1, 1):
        P_Xt[:, i] = P_Xt[:, i-1] @ Q #pi_t = pi_t-1 * Q

    return P_Xt


def P_Xt_C(Q, t):
    P_Xt = np.zeros((4, t+1))
    P_Xt[1, 0] = 1 #pi_0
    for i in range(1, t+1, 1):
        P_Xt[:, i] = P_Xt[:, i-1] @ Q #pi_t = pi_t-1 * Q

    return P_Xt

tfinal = 15
temps = np.arange(0, tfinal + 1)

P_Xt_unif = P_Xt_Unif(Q, tfinal)
P_Xt_c = P_Xt_C(Q, tfinal)
Q_15 = Q_multistep(Q, tfinal)


def Evolution(P_Xt, index):
    _, ax1 = plt.subplots()
    plt.plot(temps, P_Xt[0,:], 'cornflowerblue', label='A')
    plt.plot(temps, P_Xt[1,:], 'tab:cyan', label='C', linewidth=3)
    plt.plot(temps, P_Xt[2,:], 'tab:purple', label='G')
    plt.plot(temps, P_Xt[3,:], 'tab:pink', label='T')

    plt.title(r"Evolution de $\mathrm{\mathbb{P}(X_t=i)}$")
    plt.legend(loc='upper right')
    ax1.set_xlabel('t')
    ax1.set_ylabel(r"$\mathrm{\pi_t = \mathbb{P}(X_t=i)}$")
    ax1.spines['top'].set_visible(False)    # enlever les bords noirs en haut
    ax1.spines['right'].set_visible(False)  # enlever les bords noirs à droite

    plt.tight_layout()
    figname = "fig"+str(index)+".pdf"
    plt.savefig(figname)  # enregistrer une figure avec une variable qui change
    plt.show()


index = 1
Evolution(P_Xt_unif, index)
index += 1
Evolution(P_Xt_c, index)
index += 1
print("Q^5 = ", Q_multistep(Q, 5), "\n")
print("Q^14 = ", Q_multistep(Q, 14), "\n")
print("Q^15 = ", Q_15, "\n")


#Point 1.1 question 3 :
P_Xt_infini = P_Xt_unif[:, -1]
print("P_Xt_infini", P_Xt_unif[:,-1], 'et ', P_Xt_c[:,-1], "\n")
print("P_Xt_infini", P_Xt_infini, "\n")

# ou par les proprietes
# pi_infini est invariant = un vecteur propre à gauche de Q, associé à une valeur propre unitaire
valeurs_propres, vecteurs_propres_gauche = np.linalg.eig(Q.T)
print(valeurs_propres, vecteurs_propres_gauche)
P_Xt_infini2 = vecteurs_propres_gauche[:, 0].real / np.sum(vecteurs_propres_gauche[:, 0].real)
print('P_Xt_infini', P_Xt_infini2, "\n")


#Point 1.1 question 4 :
def parcours(Q, indice):
    aleatoire = np.random.rand()
    borne1 = Q[indice][0]
    borne2 = borne1 + Q[indice][1]
    borne3 = borne2 + Q[indice][2]
    if aleatoire < borne1:
        return 0 #A
    elif aleatoire < borne2:
        return 1 #C
    elif aleatoire < borne3:
        return 2 #G
    else:
        return 3 #T

def chaineT(T, apparition, lettre):
    lettre = parcours(Q, lettre)
    apparition[lettre] += 1
    return lettre, apparition

iterations = 1000
Tvec = np.arange(1, iterations + 1, 1)
realisation = np.zeros((iterations, 4))
apparition = np.zeros(4)
lettre = 0 #definition de lettre
for t in Tvec:
    if t == 1:
        lettre = np.random.randint(4)
        apparition[lettre] += 1
    else:
        lettre, apparition = chaineT(t, apparition, lettre)
    realisation[t - 1] = apparition/t

_, ax1 = plt.subplots()
plt.plot(Tvec, realisation[:,0], 'cornflowerblue', label='A')
plt.plot(Tvec, realisation[:,1], 'tab:cyan', label='C', linewidth=3)
plt.plot(Tvec, realisation[:,2], 'tab:purple', label='G')
plt.plot(Tvec, realisation[:,3], 'tab:pink', label='T')

plt.title("Proportion d'apparition pour T croissant")
plt.legend(loc='upper right')
ax1.set_xlabel('T')
ax1.set_ylabel("proportion d'apparition")
ax1.spines['top'].set_visible(False)    # enlever les bords noirs en haut
ax1.spines['right'].set_visible(False)  # enlever les bords noirs à droite

plt.tight_layout()
index = 3
figname = "fig"+str(index)+".pdf"
plt.savefig(figname)  # enregistrer une figure avec une variable qui change
plt.show()


#Point 1.1 question 5 :
def calculBase(seq):
    nombreBase = np.zeros(4)
    for i in range(len(seq) - 1):
        if seq[i] == 'a':
            nombreBase[0] += 1
        elif seq[i] == 'c':
            nombreBase[1] += 1
        elif seq[i] == 'g':
            nombreBase[2] += 1
        elif seq[i] == 't':
            nombreBase[3] += 1
    return nombreBase

def calculFreq(seq):
    freq = np.zeros(4)
    nombreBase = calculBase(seq)
    for i in range(4):
        if len(seq) != 0:
            freq[i] = nombreBase[i]/len(seq)
    return freq


freq1 = calculFreq(seq1)
freq2 = calculFreq(seq2)
freq3 = calculFreq(seq3)
print('freq1 = ', freq1)
print('freq2 = ', freq2)
print('freq3 = ', freq3, '\n')


def calculProbTransition(seq, i, indice):
    if seq[i+1] == 'a':
        return np.log10(Q[indice][0])
    elif seq[i+1] == 'c':
        return np.log10(Q[indice][1])
    elif seq[i+1] == 'g':
        return np.log10(Q[indice][2])
    elif seq[i+1] == 't':
        return np.log10(Q[indice][3])

def calculLogProbSeq(seq):
    logProbaSeq = 0
    if seq[0] == 'a':
        logProbaSeq += np.log10(P_Xt_infini[0])
    elif seq[0] == 'c':
        logProbaSeq += np.log10(P_Xt_infini[1])
    elif seq[0] == 'g':
        logProbaSeq += np.log10(P_Xt_infini[2])
    elif seq[0] == 't':
        logProbaSeq += np.log10(P_Xt_infini[3])
    for i in range(len(seq) - 1):
        if seq[i] == 'a':
            logProbaSeq += calculProbTransition(seq, i, 0)
        elif seq[i] == 'c':
            logProbaSeq += calculProbTransition(seq, i, 1)
        elif seq[i] == 'g':
            logProbaSeq += calculProbTransition(seq, i, 2)
        elif seq[i] == 't':
            logProbaSeq += calculProbTransition(seq, i, 3)
    return logProbaSeq

logProb1 = calculLogProbSeq(seq1)
logProb2 = calculLogProbSeq(seq2)
logProb3 = calculLogProbSeq(seq3)
print('log prob1 = ', logProb1)
print('log prob2 = ', logProb2)
print('log prob3 = ', logProb3, '\n')


#Point 1.2 question 3 :
Q_Y = np.array([[0.1, 0.05, 0.02, 0.08],[0.04, 0.11, 0.07, 0.03], [0.07, 0.05, 0.04, 0.09], [0.05, 0.06, 0.08, 0.06]])

print('Q_Y = ', Q_Y, '\n')

#Point 1.2 question 3b:
def parcoursQ(i, Q_Y, y):
    #si choix de i = 1 : Q_Y[:, y[t][1]]
    #si choix de i = 2 : Q_Y[y[t][0], :]
    if i == 1:
        Q = Q_Y[:, y[1] - 1]
    else:
        Q = Q_Y[y[0] - 1, :]
    aleatoire = np.random.rand()
    aleatoire *= (Q[0] + Q[1] + Q[2] + Q[3])
    borne1 = Q[0]
    borne2 = borne1 + Q[1]
    borne3 = borne2 + Q[2]
    if aleatoire < borne1:
        return 1
    elif aleatoire < borne2:
        return 2
    elif aleatoire < borne3:
        return 3
    return 4


def freqPaire(y, T):
    frequence = np.zeros(16)
    for t in range(T+1):
        if y[t][0] == 1 and y[t][1] == 1:
            frequence[0] += 1
        elif y[t][0] == 1 and y[t][1] == 2:
            frequence[1] += 1
        elif y[t][0] == 1 and y[t][1] == 3:
            frequence[2] += 1
        elif y[t][0] == 1 and y[t][1] == 4:
            frequence[3] += 1
        elif y[t][0] == 2 and y[t][1] == 1:
            frequence[4] += 1
        elif y[t][0] == 2 and y[t][1] == 2:
            frequence[5] += 1
        elif y[t][0] == 2 and y[t][1] == 3:
            frequence[6] += 1
        elif y[t][0] == 2 and y[t][1] == 4:
            frequence[7] += 1
        elif y[t][0] == 3 and y[t][1] == 1:
            frequence[8] += 1
        elif y[t][0] == 3 and y[t][1] == 2:
            frequence[9] += 1
        elif y[t][0] == 3 and y[t][1] == 3:
            frequence[10] += 1
        elif y[t][0] == 3 and y[t][1] == 4:
            frequence[11] += 1
        elif y[t][0] == 4 and y[t][1] == 1:
            frequence[12] += 1
        elif y[t][0] == 4 and y[t][1] == 2:
             frequence[13] += 1
        elif y[t][0] == 4 and y[t][1] == 3:
            frequence[14] += 1
        elif y[t][0] == 4 and y[t][1] == 4:
            frequence[15] += 1

    return frequence/T


def find_y_Unif(y_1, y_2, T):
    y_Unif = np.zeros((T+1, 2), dtype=int)
    y_Unif[0][0] = y_1
    y_Unif[0][1] = y_2

    for t in range(T):
        y_Unif[t+1] = y_Unif[t]
        random = np.random.rand()
        i = 2
        if random < 0.5:
            i = 1
        y_Unif[t+1][i-1] = parcoursQ(i, Q_Y, y_Unif[t])

    return y_Unif

def find_y_alterne(y_1, y_2, i, T):
    y_alterne = np.zeros((T+1, 2), dtype=int)
    y_alterne[0][0] = y_1
    y_alterne[0][1] = y_2
    for t in range(T):
        y_alterne[t+1] = y_alterne[t]
        if i == 1:
            i = 2
        elif i == 2:
            i = 1
        y_alterne[t+1][i-1] = parcoursQ(i, Q_Y, y_alterne[t])

    return y_alterne

def plotBarFrequence(frequence_100, frequence_5000, frequence_reference, index):
    Axe_x_frequence = ["1;1", "1;2", "1;3", "1;4", "2;1", "2;2", "2;3", "2;4", "3;1", "3;2", "3;3", "3;4", "4;1", "4;2", "4;3", "4;4"]
    # Position sur l'axe des x pour chaque étiquette
    position = np.arange(len(Axe_x_frequence))
    # Largeur des barres
    largeur = .3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(position - largeur, frequence_100, largeur, color='tab:cyan', label="100")
    ax.bar(position, frequence_5000, largeur, color='tab:purple', label="5000")
    ax.bar(position + largeur, frequence_reference, largeur, color='tab:pink', label="référence")

    ax.set_xticks(position)
    ax.set_xticklabels(Axe_x_frequence)
    plt.xlabel("états (Y$_1$; Y$_2$)")
    plt.ylabel("fréquences d'apparition")
    plt.legend(loc="best")
    plt.title("Fréquences d'apparition de chaque état")
    ax.spines['top'].set_visible(False)    # enlever les bords noirs en haut
    ax.spines['right'].set_visible(False)  # enlever les bords noirs à droite
    figname = "fig"+str(index)+".pdf"
    plt.savefig(figname, bbox_inches='tight')  # enregistrer une figure avec une variable qui change
    plt.show()


random1 = np.random.rand()
random2 = np.random.rand()
y_1 = 4
y_2 = 4
if random1 < 0.25:
    y_1 = 1
elif random1 < 0.5:
    y_1 = 2
elif random1 < 0.75:
    y_1 = 3
    
if random2 < 0.25:
    y_2 = 1
elif random2 < 0.5:
    y_2 = 2
elif random2 < 0.75:
    y_2 = 3

T = 100
y_Unif_100 = find_y_Unif(y_1, y_2, T)
frequence_Unif_100 = freqPaire(y_Unif_100, T)
T = 5000
y_Unif_5000 = find_y_Unif(y_1, y_2, T)
frequence_Unif_5000 = freqPaire(y_Unif_5000, T)

frequence_reference = np.reshape(Q_Y, 16)
index += 1
plotBarFrequence(frequence_Unif_100, frequence_Unif_5000, frequence_reference, index)

#Point 1.2 question 3c :
random = np.random.rand()
i = 2
if random < 0.5:
    i = 1

T = 100
y_alterne_100 = find_y_alterne(y_1, y_2, i, T)
frequence_alterne_100 = freqPaire(y_alterne_100, T)
T = 5000
y_alterne_5000 = find_y_alterne(y_1, y_2, i, T)
frequence_alterne_5000 = freqPaire(y_alterne_5000, T)

index += 1
plotBarFrequence(frequence_alterne_100, frequence_alterne_5000, frequence_reference, index)
