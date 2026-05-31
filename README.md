# Child Well-being

Analysis of child well-being and public expenditure across 16 European OECD countries using **Partially Ordered Sets (POSets)**.

Rather than collapsing multi-dimensional indicators into a single composite index, the project preserves incomparability between countries and studies their structural relationships through posetic methods. The analysis covers two observation years (**2015** and **2018**) and is organised into two parallel analytical domains: **child outcomes & circumstances** (20 indicators, Groups A–B–C well-being) and **public expenditure & policies** (7 indicators, Group C spending).


## Authors

- [Brasca Federica](https://github.com/federicabrasca)
- [Caglio Simone](https://github.com/SimoneFisico)
- [Maggioni Pietro](https://github.com/pietromaggioni)
- [Manduca Marco](https://github.com/MarcoManduca)


## Data Source

[OECD Child Well-being Explorer](https://data-explorer.oecd.org/vis?lc=en&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_CWB%40DF_CWB&df[ag]=OECD.WISE.CWB&dq=..&to[TIME_PERIOD]=false&pd=%2C&lb=nm&vw=tb)

Extracted via SDMX/CSV API (April 2025). 27 indicators across 16 European OECD countries for years 2015 and 2018.

**Countries:** AUT, CZE, EST, ESP, FRA, GBR, HUN, ITA, LTU, LUX, LVA, POL, PRT, SVK, SVN, SWE


## Methodology

### Partially Ordered Sets (POSets)

A POSet models a set of countries under a **component-wise dominance relation**: country $A$ dominates country $B$ if $A$ performs at least as well as $B$ on **every** indicator. Countries that differ in at least one direction are *incomparable* — a distinction that composite indexes erase.

Null values are not imputed. Instead, they are treated as **structural uncertainty**: each unit with missing values occupies an interval $[\text{lo}, \text{hi}]$ in the hyperlattice, supporting `certain`, `possible`, and `certain_or_possible` dominance modes.

The `poset/` library is a Python port of the R package `poseticDataAnalysis`.

> Fattore M., Alaimo L.S. (2023). *A partial order toolbox for building synthetic indicators of sustainability with ordinal data.* Socio-Economic Planning Sciences. [doi:10.1016/j.seps.2023.101623](https://doi.org/10.1016/j.seps.2023.101623)

> Fattore M., De Capitani L., Avellone A., Suardi A. (2024). *A fuzzy posetic toolbox for multi-criteria evaluation on ordinal data systems.* Annals of Operations Research. [doi:10.1007/s10479-024-06352-3](https://doi.org/10.1007/s10479-024-06352-3)

---

### MRP Cascade Aggregation

Indicators are aggregated into macro-dimensions through a two-level **Mutual Ranking Probability (MRP) cascade** (`scripts/transformer.py`). The MRP score of unit $i$ is its average dominance probability over all pairwise comparisons:

$$\text{MRP}_{score}(i) = \frac{1}{n-1} \sum_{j \neq i} P_{\sigma \sim \text{LE}(P)}[\sigma(i) > \sigma(j)]$$

The cascade proceeds in two levels:

1. **Level 1 (`cascade_aggregate`)** — a sub-POSet is built for each subdimension group and MRP scores are computed. Single-indicator groups are passed through directly. Output: one continuous score per unit per subdimension.

2. **Level 2 (`cascade_level2`)** — subdimension MRP scores are discretised and a second POSet is built per macro-dimension. Output: the `indicators_macro_dim` dataset variants used in notebooks 050–090.

All indicators are normalised so that **higher values always represent better outcomes** before entering the cascade. Negative-direction indicators (e.g. poverty rate, infant mortality) are inverted during the normalisation step.

---

### Minimum Dominating Set

The **Minimum Dominating Set (MDS)** is the smallest subset $D \subseteq P$ such that every element is either in $D$ or dominated by at least one element of $D$:

$$D \subseteq P : \forall\, p \in P,\; p \in D \;\lor\; \exists\, d \in D : d \succ p$$

In this project the MDS corresponds to the **maximal elements** — countries that no other country consistently outperforms. A small MDS (1–2 countries) indicates a clear dominance hierarchy; a large MDS indicates high structural incomparability. Notebook 070 also provides a **bottleneck analysis** that identifies, for any pair of countries, the specific indicators blocking certain dominance.

---

### Posetic Separation

$\text{Sep}(i,j)$ measures how far apart two countries rank on average across all linear extensions of the poset:

$$\text{Sep}(i,j) = \frac{\mathbb{E}_{\sigma \sim \text{LE}(P)}\bigl[|\sigma(i) - \sigma(j)|\bigr]}{n - 1} \in [0,1]$$

A value near 1 means $i$ and $j$ consistently occupy opposite ends; near 0 means they rank similarly. The expectation is approximated via **Bubley-Dyer MCMC** sampling (50 000 linear extensions per poset).

- **Notebook 080** — pairwise separation matrices per dataset and year.
- **Notebook 090** — separation used as dissimilarity input to Classical MDS.

---

### cMDS Country Projection

Countries are embedded in 2D/3D space via **Classical Multi-Dimensional Scaling (cMDS / PCoA)** using the posetic separation matrix. Embeddings for 2015 and 2018 are independently computed and then aligned with **Procrustes analysis** (rotation + reflection, no scaling) so that arrows between the same country across years represent genuine positional shifts in the dominance space.

---

## Project Structure

### Analysis Pipeline

| Notebook | Description |
|---|---|
| [`010_data_extractor.ipynb`](notebooks/010_data_extractor.ipynb) | Data extraction from the OECD API → `data/010_child_well_being.parquet` |
| [`020_data_exploration.ipynb`](notebooks/020_data_exploration.ipynb) | Exploratory data analysis and correlation study |
| [`030_data_transformer.ipynb`](notebooks/030_data_transformer.ipynb) | Direction-corrected normalisation and ordinal discretisation of all 27 indicators |
| [`040_data_partitioner.ipynb`](notebooks/040_data_partitioner.ipynb) | MRP cascade aggregation; partition by year and domain → all `040_*.parquet` files |
| [`050_Poset_creator.ipynb`](notebooks/050_Poset_creator.ipynb) | Build POSets from all partitions → `data/050_posets*.pkl` |
| [`054_Poset_check.Rmd`](notebooks/054_Poset_check.Rmd) | R reference implementation (`poseticDataAnalysis`) for cross-language validation |
| [`055_Poset_check.ipynb`](notebooks/055_Poset_check.ipynb) | Python-side cross-language validation against the R reference |
| [`060_Poset_analysis.ipynb`](notebooks/060_Poset_analysis.ipynb) | MRP-based country rankings and macro-dimension profile heatmaps |
| [`070_MDS_visualization.ipynb`](notebooks/070_MDS_visualization.ipynb) | Minimum Dominating Set analysis, Hasse diagrams, and bottleneck analysis |
| [`080_dominance_matrices.ipynb`](notebooks/080_dominance_matrices.ipynb) | Certain / possible dominance heatmaps and posetic separation matrices |
| [`090_MDS_projection.ipynb`](notebooks/090_MDS_projection.ipynb) | cMDS + Procrustes projection: country trajectories 2015 → 2018 |

> Notebooks must be run in order (010 → 090). The 030 → 040 step produces all intermediate parquet files that the analysis notebooks depend on.

### Discretization Variants

| Dataset | Levels | Domain |
|---|---|---|
| `indicators_macro_dim` | 3, 4, 5 | Child outcome macro-dimensions |
| `indicators_dim_discrete` | 3, 4 | Child outcome subdimensions |
| `public_expenditure_dim_discrete` | 3, 4 | Public expenditure by category |

### `scripts/` Module

| Function | Description |
|---|---|
| `normalize_minmax` | Direction-corrected min-max normalisation (inverts negative indicators) |
| `discretize` | Ordinal discretisation of normalised columns (quantile or equal-width) |
| `cascade_aggregate` | Level 1 MRP cascade: indicators → subdimension scores |
| `cascade_level2` | Level 2 MRP cascade: subdimension scores → macro-dimension scores |

### `poset/` Library

Custom Python library implementing the posetic toolbox. Exposed via a clean public API in `__init__.py`.

| Module | Content |
|---|---|
| `poset.py` | Core data structures: `POSet`, `LinearPOSet`, `BinaryVariablePOSet` |
| `poset_ops.py` | Algebraic operations: product, dual, sum, lifting, crown, fence |
| `relations.py` | Reflexivity / transitivity / antisymmetry checks and closures (Floyd-Warshall) |
| `poset_query.py` | Element queries: minimals, maximals, upset/downset, covers, meet/join |
| `linear_extensions.py` | Exact LE generator + Bubley-Dyer MCMC sampler |
| `mrp.py` | Mutual Ranking Probabilities (exact + Bubley-Dyer) |
| `separation.py` | Separation scores (exact + Bubley-Dyer) |
| `evaluation.py` | Function averaging over linear extensions |
| `dominance.py` | BLS dominance matrix |
| `fuzzy.py` | Fuzzy in-betweenness and separation (MinMax + Probabilistic) |
| `embedding.py` | Bidimensional embedding (PARSEC-style, optimal permutation search) |
| `from_polars.py` | Build POSet from Polars DataFrames with null-as-uncertainty intervals |

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11+. The `poset/` library requires no installation — notebooks import it via `sys.path.insert(0, '..')`.

---

## Indicators

27 indicators selected from the OECD CWB framework, split into two analytical domains. All negative-direction indicators are inverted during normalisation so that **higher ordinal level always means better outcome**.

### Domain 1 — Child Well-being (20 indicators)

| Code | Indicator | Dir | Subdimension |
|---|---|---|---|
| A1_2 | Children experiencing severe housing deprivation | − | A1 · Material conditions |
| A2_1 | Infant mortality rate | − | A2 · Health |
| A3_3 | Top performers in PISA (level 5–6, at least one subject) | + | A3 · Education |
| A3_4 | Students expecting to complete tertiary education | + | A3 · Education |
| A3_5 | Youth NEET rate (15–29 yrs) | − | A3 · Education |
| A4_6 | Students reporting high life satisfaction (score 9–10) | + | A4 · Subjective well-being |
| B1_1 | Children in relative income poverty (50% median threshold) | − | B1 · Family income |
| B1_5 | Parents strongly encourage self-confidence (PISA) | + | B1 · Family income |
| B2_4 | Students experiencing bullying at school | − | B2 · School & early childhood |
| B2_5 | Students feeling they belong at school | + | B2 · School & early childhood |
| B3_5 | Children in households reporting local crime / violence | − | B3 · Safety |
| B4_3 | Students believing Internet is a great info resource | + | B4 · Digital environment |
| C1_2 | Poverty reduction via taxes and transfers (pp) | + | C1 · Family spending |
| C1_3 | Guaranteed minimum income — jobless couple, 2 children | + | C1 · Family spending |
| C1_4 | Paid maternity / parental leave available to mothers (weeks) | + | C1 · Family spending |
| C1_5 | Paid paternity / parental leave reserved for fathers (weeks) | + | C1 · Family spending |
| C3_2 | DPT vaccination rate (3 doses, under 1 yr) | + | C3 · Public health |
| C3_3 | Measles vaccination rate (at least 1 dose, under 1 yr) | + | C3 · Public health |
| C4_2 | Net childcare costs for parents (low-earning couple, 2 children) | − | C4 · Education & childcare |
| C4_6 | Student-to-teacher ratio in secondary education | − | C4 · Education & childcare |

### Domain 2 — Public Expenditure (7 indicators)

| Code | Indicator | Dir | Category |
|---|---|---|---|
| C2_1 | Gov. expenditure on housing & community amenities per person | + | C2 · Housing & culture |
| C2_2 | Gov. expenditure on recreation, culture & religion per person | + | C2 · Housing & culture |
| C2_3 | Gov. expenditure on housing social protection per person | + | C2 · Housing & culture |
| C3_1 | Gov. & compulsory health insurance expenditure per person | + | C3 · Public health |
| C4_4 | Public expenditure on primary & secondary education per FTE student | + | C4 · Education & childcare |
| C4_5 | Public expenditure on ancillary education services per FTE student | + | C4 · Education & childcare |
| C5_1 | Gov. expenditure on environment protection per person | + | C5 · Environment |
