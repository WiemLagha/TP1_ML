from ast import Index
import os
import pandas
import numpy
#importation de la fonction apriori à partir de la bibliothèque mlxtend
from mlxtend.frequent_patterns import apriori

#implémentation de la fonction is_inclus
def is_inclus(x,items):
 return items.issubset(x)

#changement de dossier
os.chdir("C:/Users/Wiem/Desktop/TP1_ML")

#Création du dataFrame
#Les valeurs sont délimités par une tabulation
#La ligne à utiliser comme nom de colonne est la ligne 0
D = pandas.read_table("market_basket.txt",delimiter="\t",header=0)

#Affichage des 10 premières lignes
#print(D.head(10))

#Affichage des dimension du dataframe
#print(D.shape)

#Construction de la table binaire
TC = pandas.crosstab(D.ID,D.Product)

#Affichage des 30 premières transaction et des 3 premiers produits
print(TC.iloc[:30,:3])


#Extraction des itemsets frequents
freq_itemsets = apriori(TC,min_support=0.025,max_len=4,use_colnames=True)
#Les itemsets fréquents seront stockés dans une structure
# de type ‘’pandas / DataFrame’’
type(freq_itemsets)

#affichage des 15 premiers itemsets
print(freq_itemsets.head(15))

#recherche des index des itemsets qui contienneent Aspirin
id = numpy.where(freq_itemsets.itemsets.apply(is_inclus,items={'Aspirin'}))
#affichage des itemsets corresp.
for i in id:
   print(freq_itemsets.loc[i])

print(freq_itemsets[freq_itemsets['itemsets'].eq('Aspirin')])

#itemsets contenant Aspirin et Eggs
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Aspirin','Eggs'})])

#fonction de calcul des règles
from mlxtend.frequent_patterns import association_rules
from ast import Index

#génération des règles à partir des itemsets fréquents
regles = association_rules(freq_itemsets,metric="confidence",min_threshold=0.75)
#type de l'objet renvoyé
print(type(regles))
#dimension
print(regles.shape)

#liste des colonnes
print(regles.columns)

#5 "premières" règles
print(regles.iloc[:5,:])

myRegles = regles.loc[:,['antecedents','consequents','lift']]
#affichage des règles avec un LIFT supérieur ou égal à 7
print(myRegles[myRegles['lift'].ge(7.0)])

#filtrer les règles menant au conséquent {‘2pct_milk’}
print(myRegles[myRegles['consequents'].eq({'2pct_Milk'})])
