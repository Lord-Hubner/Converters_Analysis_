import pandas as pd
#from sklearn.model_selection import

#luad = pd.read_csv("PRAD_features.csv", index_col=0)

#print(luad.shape)

##luad_gene_counts = luad["gene_symbol"].value_counts()

##luad_selected = luad[luad["gene_symbol"].isin(luad_gene_counts[luad_gene_counts>1].index)] 

#luad = luad.drop(columns="gene_symbol")
#luad = luad[luad["padj"].isna() == 0]

#arenull = luad.isnull().sum()
#print(arenull)

#luad.to_csv("PRAD_features_nogene.csv", index=True)


table = pd.read_csv('KIRC_features_nogene.csv', index_col=0)

y = table["label"]
X = table.drop(columns="label")


y.to_csv("label_KIRC")
X.to_csv("data_KIRC")

