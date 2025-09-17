# Model-based enhancer design code

The following scripts (coming soon!) contain code to perform various design tasks:

- `dhs64_single_fsp.py`: Design enhancers for activity specific to one of the [DHS64-modeled biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) against all others. Uses [Fast SeqProp](https://doi.org/10.1186/s12859-021-04437-5).
- `dhs64_single_den.py`: Train a [Deep Exploration Network](https://doi.org/10.1016/j.cels.2020.05.007) to generate enhancers with activity specific to one of the DHS64-modeled biosamples against all others.
- `dhs64_single_den_generate_only.py`: Use a pretrained Deep Exploration Network to generate new enhancer sequences.
- `dhs64_single_tunable.py`: Design enhancers with tunable activity specific to one of the DHS64-modeled biosamples. Uses Fast SeqProp.
- `dhs64_double.py`: Design enhancers with activity specific to two of the DHS64-modeled biosamples against all others. Uses Fast SeqProp.
- `dhs64_double.py`: Design enhancers with activity specific to three of the DHS64-modeled biosamples against all others. Uses Fast SeqProp.
- `dhs733_single_fsp.py`: Design enhancers for activity specific to any DNase I Index biosample against all others. Only considers non-redundant biosamples (i.e. with non-repeated biosample names), resulting in 261 possible targets. Uses Fast SeqProp.
- `analysis.ipynb`: Notebook to perform basic analysis of designed sequences, including sequence quality metrics, motif alignments, and cell type contribution scores.