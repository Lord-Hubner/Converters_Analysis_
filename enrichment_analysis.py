import os
import pandas as pd
from gprofiler import GProfiler
import matplotlib.pyplot as plt 

def get_gene_ids(df, gene_column=None):
    if gene_column is not None and gene_column in df.columns:
        genes = df[gene_column].astype(str)
    else:
        genes = df.index.astype(str)

    # Remove Ensembl version suffix if present:
    # ENSG000001234.5 -> ENSG000001234
    genes = genes.str.replace(r"\..*$", "", regex=True)

    return genes


def run_gprofiler_go(
    file_path,
    dataset_name,
    gene_column=None,
    log2fc_col="log2FC",
    padj_col="padj",
    label_col=None,
    log2fc_cutoff=1,
    padj_cutoff=0.05,
    output_dir="GO_results_python"
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path, index_col=0)

    genes = get_gene_ids(df, gene_column)
    df = df.copy()
    df["gene_for_go"] = genes

    print(df)

    print("------------------")
    print(genes)
    df = df.dropna(subset=["gene_for_go", log2fc_col, padj_col])


    # Background universe: all genes that survived your filtering/table construction.
    background = df["gene_for_go"].dropna().unique().tolist()

    if label_col is not None and label_col in df.columns:
        up_genes = df.loc[df[label_col].isin(["UR", "Up", "Upregulated"]), "gene_for_go"].unique().tolist()
        down_genes = df.loc[df[label_col].isin(["DR", "Down", "Downregulated"]), "gene_for_go"].unique().tolist()
    else:
        up_genes = df.loc[
            (df[padj_col] < padj_cutoff) & (df[log2fc_col] >= log2fc_cutoff),
            "gene_for_go"
        ].unique().tolist()

        down_genes = df.loc[
            (df[padj_col] < padj_cutoff) & (df[log2fc_col] <= -log2fc_cutoff),
            "gene_for_go"
        ].unique().tolist()

    gp = GProfiler(return_dataframe=True)

    gene_sets = {
        "Upregulated": up_genes,
        "Downregulated": down_genes
    }

    for regulation, gene_list in gene_sets.items():
        print(f"{dataset_name} - {regulation}: {len(gene_list)} genes")

        if len(gene_list) < 10:
            print(f"Skipping {dataset_name} - {regulation}: too few genes.")
            continue

        result = gp.profile(
            organism="hsapiens",
            query=gene_list,
            sources=["GO:BP"],
            user_threshold=0.05,
            significance_threshold_method="fdr",
            background=background
        )

        output_file = os.path.join(
            output_dir,
            f"{dataset_name}_{regulation}_GO_BP_gprofiler.csv"
        )

        result.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


#run_gprofiler_go("KIRC_features.csv", "KIRC")
#run_gprofiler_go("LUAD_features.csv", "LUAD")
#run_gprofiler_go("PRAD_features.csv", "PRAD")

def plot_go_results(csv_file, title, output_png, top_n=15):
    df = pd.read_csv(csv_file)

    if df.empty:
        print(f"No enriched terms in {csv_file}")
        return

    df = df.sort_values("p_value").head(top_n).copy()
    df["minus_log10_p"] = -df["p_value"].apply(lambda x: __import__("math").log10(x))

    for a in df["name"]:
        if len(a)>50:
            print(a)
        
    df["name"] = df["name"].apply(lambda x: "".join(word[0].upper() for word in x.split()) if len(x) > 50 else x)

    plt.figure(figsize=(12, 6))
    plt.barh(df["name"][::-1], df["minus_log10_p"][::-1])
    plt.subplots_adjust(left=0.3)
    plt.yticks(fontsize=9)
    plt.xlabel("-log10(adjusted p-value)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=600)
    plt.close()

plot_go_results(
    "GO_results_python/LUAD_Downregulated_GO_BP_gprofiler.csv",
    "LUAD - Downregulated GO Biological Process",
    "GO_results_python/LUAD_Downregulated_GO_BP.png"
)