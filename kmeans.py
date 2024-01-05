# Aout 22
# Kmeans pour TC1 BE1
# 2 clusters

# Ci-dessous, j'ai testé la MM chose (sans les plots) avec le packages scipy qui a "kmeans2" ...
# voir la fonction "tester_choix_init_kmeans_plusplus_et_Kmeans_de_scipy"

import random, math
import numpy as np
import matplotlib.pyplot as plt

def dist_euclid(a,b) :
    x1,y1=a
    x2,y2=b
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def dist_avec_norme(a,b) :
    from scipy.linalg import norm
    norm(a - b)
#---------------------------------------------------------
def centre_de_nuage(Lst) :
    x_m =sum([x for (x,y) in Lst])/len(Lst)
    y_m =sum([y for (x,y) in Lst])/len(Lst)
    return (x_m, y_m)
#---------------------------------------------------------
def K_means_AVEC_option_plots_intermediaires(Lst, avec_plots=True) :
    m_1 = Lst[0]
    for i in range(1,NB_points) :
        m_2 = Lst[i]  
        if m_1 != m_2 :  break
    
    if avec_plots :
        print("m1 :", m_1, "m2 : ", m_2)

    fini = False
    old_m1 = m_1; old_m2 = m_2
    L_1=[]; L_2=[]
    iteration=1
    while not  fini :
        L_1=[]; L_2=[] # Important
        for point in Lst :
            if dist_euclid(point, m_1) < dist_euclid(point, m_2) : L_1.append(point)
            else : L_2.append(point)
        m_1 = centre_de_nuage(L_1) # m_1 = la moyenne de L_1
        # m_1 ne sera pas forcément un des points de L_1
        m_2 = centre_de_nuage(L_2)
        #diff_x=m_1[0]
        if dist_euclid(m_1 , old_m1) < epsilon or dist_euclid(m_2 , old_m2) < epsilon : fini = True
        old_m1 = m_1; old_m2 = m_2
        Lst = L_1 + L_2
    
        if avec_plots :
            # 'ro' : red 'o' (round), 'go'b= green 'o', 'bs' = blue square, 'ys' = yello square 
            # "og" = 'o' green,  'r+' : des croix rouges, 
            # '^b' : flèche bleue, "-k" = ligne noire, "--c" = ligne cyan pointillé
            # "-vy" =  ligne yellow triangle à l'envers
            plt.plot([x for (x,y) in L_1], [y for (x,y) in L_1], 'r+')
            plt.plot(m_1[0], m_1[1], 'go')
            #plt.axis([0, 6, 0, 20])
            #plt.show()
            plt.plot([x for (x,y) in L_2], [y for (x,y) in L_2], 'b+')
            plt.plot(m_2[0], m_2[1], 'ys')
            #plt.axis([0, 6, 0, 20])
            plt.show()
            #print("Itération ", iteration, " : ", m_1, L_1, m_2, L_2)
            print("Itération ", iteration, " : ", m_1,  m_2)
            #input(" ? ")
            iteration+=1
    # Fin While
    return m_1, L_1, m_2, L_2
#---------------------------------------------------------
# Err au sein de chaque cluster
# On applique sqrt à la somme des err des 2 clusters pour manipuler des nbrs plus petits
def errSSE(m_1, L_1, m_2, L_2) : # Only op==pour 2 clusters
    part1= sum([dist_euclid(m_1, p) for p in L_1])
    part2= sum([dist_euclid(m_2, p) for p in L_2])
    return math.sqrt(part1+part2)
#---------------------------------------------------------
def errSSD(m_1, m_2) : # Only pour 2 clusters : simple err entre les centres
    return dist_euclid(m_1, m_2)

#========================================================
# MM chose (sans les plots) avec le packages scipy qui a "kmeans2" ...
from scipy.cluster.vq import kmeans2
#from scipy import random, array
from scipy.linalg import norm
from functools import reduce

# Choix des centres selon Kmeans++
def kinit(X, k): # X : échantillons, k = nb clusters
    'init k seeds according to kmeans++'
    n = X.shape[0] # nb points d'un array de numpy

    'choose the 1st seed randomly, and store D(x)^2 in D[]'
    centers = [X[random.randint(1,n)]]
    D = [norm(x - centers[0]) ** 2 for x in X] # dist euclidienne

    for _ in range(k - 1):
        bestDsum = bestIdx = -1

        for i in range(n):
            'Dsum = sum_{x in X} min(D(x)^2,||x-xi||^2)'
            Dsum = reduce(lambda x, y:x + y,
                          (min(D[j], norm(X[j] - X[i]) ** 2) for j in range(n)))

            if bestDsum < 0 or Dsum < bestDsum:
                bestDsum, bestIdx = Dsum, i

        centers.append (X[bestIdx])
        D = [min(D[i], norm(X[i] - X[bestIdx]) ** 2) for i in range(n)]

    return array (centers)

# Pour tester cette fonction :
# Créez comme ci-dessous la liste Lst puis
# ZZ :  kmeans2 demande des réels
def tester_choix_init_kmeans_plusplus_et_Kmeans_de_scipy(Lst, k) :
    k=2
    X=np.array(Lst)
    # return kinit(X,k) # renvoie les k centre selon Kmeans++
    return kmeans2(X, kinit(X, k), minit='points')

# Depuis main, après la création de Lst, j'ai testé comme ceci :
    # NB : kmeans2 de scipy demandes des points réels d'ou la conversion de Lst ci-après :
    print(tester_choix_init_kmeans_plusplus_et_Kmeans_de_scipy([(1.0*x,1.0*y) for (x,y) in Lst], 2)) 
    # A donné (pour 10 points) : les 2 centres et les clusters des 10 points de test
    #(array([[5942.55555556, 6664.22222222],
    #   [  67.        ,  893.        ]]), 
    # array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int32)) <<- les clusters des 10 points (clusters 0 ou 1)
# -------------------------------------------------------
if __name__ == "__main__" :
    Max=15000
    NB_points=300
    epsilon=0.0001

    # Mon tirage a lieu sur la moitié inférieure gauche du plan et le 2/3 .. 3/4 sup droit
    # Mais en fait, j'ai rechangé ! je prend la moitié des points dans le premier 2/3 du plan et l'autre moitié 
    # dans 1/2.. 3/4 du plan
    Lst1 = [(random.randint(1,2*Max//3), random.randint(1,2*Max//3)) for i in range(NB_points//2)]
    Lst2 = [(random.randint(Max//3, 3*Max//4), random.randint(Max//3, 3*Max//4)) for i in range(NB_points//2)]
    Lst=Lst1 + Lst2

    # Lst = [(random.randint(1,Max), random.randint(1,Max)) for i in range(NB_points)]
    # se limiter à [(0,0) .. (100,100)]
    m_1, L_1, m_2, L_2 = K_means_AVEC_option_plots_intermediaires(Lst)
    #print("Resultats : cluster 1 : ", (m_1, L_1), "cluster 2 : ", (m_2, L_2 ))
    print(f"Resultats : centre 1 : {m_1}  avec {len(L_1)} élés et centre 2 : {m_2} avec  {len(L_2)} élés")
    print(f"Err SSE = {errSSE(m_1, L_1, m_2, L_2)}, SSD= {errSSD(m_1, m_2)}")

    # ======================
    # On va maintenant itérer sur kmeans pour trouver le meilleurs 2-clustering
    print("-"*80)
    NB_essais=10
    lst_resultats=[]
    for i in range(NB_essais) :
        random.shuffle(Lst) # Pour ne pas avoir les mm résultats dans chaque itération
        m_1, L_1, m_2, L_2 = K_means_AVEC_option_plots_intermediaires(Lst, avec_plots=False)
        e1,e2 = errSSE(m_1, L_1, m_2, L_2) , errSSD(m_1, m_2)
        lst_resultats.append((m_1, L_1, m_2, L_2, e1, e2))

    # On trie sur SSD (e1) puis l'inverse de e2 (car on minimise e1 tout en maximisant e2)
    # Le tri alieu sur un coupe (e1, 1/e2). Est-ce le mieux ?
    # Attenton : les 2 errs ne sont pas du mm ordre : e2 est en géné 10 fois plus grande que e1 pour ces données
    lst_triee = sorted(lst_resultats, key=lambda x   : (x[4], 1/e2)) # x de la forme : m1,l1,m2,l2,e1,e2
    
    # On affiche les résultats des itérations. On a trié selon l'erreur SSd (inter classes)
    print( "I \t\t m1  \t \t\t  m2    \t   e1   \t   e2")
    for i in range(NB_essais) :
        m1,l1,m2,l2,e1,e2=lst_triee[i]
        v1_x = round(m1[0],2); v1_y = round(m1[1],2) # juste pour affichage
        v2_x = round(m2[0],2); v2_y = round(m2[1],2)
        print(i, "\t", (v1_x,v1_y),"\t", (v2_x,v2_y), "\t",round(e1,2), "\t",round(e2,2))        

"""
## On a la trace d'un exécution avec des plots puis, après e ligne des '--', on a 10 itérations (sans trace ni plot)  
alex@Begonia:~/ECL-ALL/ECL-22-23/1A-22-23/TC1-BE1-22-23-changement-partie-eleves$ python3 kmeans.py 
m1 : (6881, 5209) m2 :  (6831, 8641)
Itération  1  :  (5633.433333333333, 3938.4333333333334) (7193.14, 8878.746666666666)
Itération  2  :  (4876.7131782945735, 3586.2403100775196) (7572.456140350877, 8537.730994152047)
Itération  3  :  (4400.318965517241, 3446.939655172414) (7682.33152173913, 8275.717391304348)
Itération  4  :  (4207.419642857143, 3438.1071428571427) (7727.420212765957, 8178.239361702128)
Itération  5  :  (4174.892857142857, 3461.5446428571427) (7746.797872340426, 8164.276595744681)
Itération  6  :  (4177.371681415929, 3492.353982300885) (7764.401069518716, 8170.807486631016)
Itération  7  :  (4181.850877192983, 3521.2631578947367) (7780.940860215053, 8178.241935483871)
Itération  8  :  (4181.947826086956, 3553.3739130434783) (7800.335135135135, 8183.454054054054)
Itération  9  :  (4154.310344827586, 3606.689655172414) (7837.423913043478, 8175.005434782609)
Itération  10  :  (4154.310344827586, 3606.689655172414) (7837.423913043478, 8175.005434782609)
Resultats : centre 1 : (4154.310344827586, 3606.689655172414)  avec 116 élés et centre 2 : (7837.423913043478, 8175.005434782609) avec  184 élés
Err SSE = 901.8244964471329, SSD= 5868.120194628899
--------------------------------------------------------------------------------
I                m1                       m2               e1              e2
0        (4106.43, 3869.16)      (8016.36, 8173.28)      901.8   5814.89
1        (4106.43, 3869.16)      (8016.36, 8173.28)      901.8   5814.89
2        (4049.4, 3857.28)       (8011.22, 8133.22)      901.96          5829.2
3        (8011.22, 8133.22)      (4049.4, 3857.28)       901.96          5829.2
4        (4049.4, 3857.28)       (8011.22, 8133.22)      901.96          5829.2
5        (4049.4, 3857.28)       (8011.22, 8133.22)      901.96          5829.2
6        (4218.57, 3265.08)      (7630.05, 8151.36)      902.41          5959.35
7        (4081.23, 3329.1)       (7687.51, 8091.2)       902.53          5973.51
8        (7687.51, 8091.2)       (4081.23, 3329.1)       902.53          5973.51
9        (7516.3, 8044.68)       (4104.93, 2984.6)       903.15          6102.61


On a trié sur SSE (somme des errs dans chaque cluster)  et l'inverse de SSD (err entre le scentres des clusters).
"""
