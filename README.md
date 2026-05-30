# Child Well-being

Analysis of child well-being indicators across OECD countries using **Partially Ordered Sets (POSets)**.  
Rather than collapsing multi-dimensional indicators into a single composite index, the project preserves incomparability between countries and studies their structural relationships through posetic methods.

---

## Authors

- [Brasca Federica](https://github.com/federicabrasca)
- [Caglio Simone](https://github.com/SimoneFisico)
- [Maggioni Pietro](https://github.com/pietromaggioni)
- [Manduca Marco](https://github.com/MarcoManduca)

---

## Data Source

[OECD Child Well-being Explorer](https://data-explorer.oecd.org/vis?lc=en&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_CWB%40DF_CWB&df[ag]=OECD.WISE.CWB&dq=..&to[TIME_PERIOD]=false&pd=%2C&lb=nm&vw=tb)

Two observation years: **2015** and **2018**.  
~30 indicators covering child outcomes, circumstances, and policies across 16 European OECD countries.

---

## Methodology

### Partially Ordered Sets (POSets)

A POSet models a set of countries under a **component-wise dominance relation**: country $A$ dominates country $B$ if $A$ performs at least as well as $B$ on **every** indicator. Countries that differ in at least one direction are *incomparable*, a distinction that composite indexes erase.

Null values in the data are not imputed. Instead, they are treated as **structural uncertainty**: each unit with missing values occupies an interval $[\text{lo}, \text{hi}]$ in the hyperlattice, supporting `certain`, `possible`, and `certain_or_possible` dominance modes.

The posetic library is a Python port of the R package `poseticDataAnalysis`

References:
> Fattore M., Alaimo L.S. (2023).  
> *A partial order toolbox for building synthetic indicators of sustainability with ordinal data*  
> Socio-Economic Planning Sciences. [doi:10.1016/j.seps.2023.101623](https://doi.org/10.1016/j.seps.2023.101623)

> Fattore M., De Capitani L., Avellone A., Suardi A. (2024).  
> *A fuzzy posetic toolbox for multi-criteria evaluation on ordinal data systems.*  
> Annals of Operations Research. [doi:10.1007/s10479-024-06352-3](https://doi.org/10.1007/s10479-024-06352-3)

---

### MRP Cascade Aggregation

Individual indicators are aggregated into macro-dimensions through a two-level **Mutual Ranking Probability (MRP) cascade**, implemented in `scripts/transformer.py`.

**MRP score** for a unit $i$ is defined as its average dominance probability across all pairwise comparisons:

$$\text{MRP}_{score}(i) = \frac{1}{n-1} \sum_{j \neq i} P_{\sigma \sim \text{LE}(P)}[\sigma(i) > \sigma(j)]$$

The cascade proceeds in two levels:

1. **Level 1 (`cascade_aggregate`)** — for each subdimension group, a sub-POSet is built on the individual indicators and MRP scores are computed. Groups with a single indicator are passed through directly. The result is one continuous score per unit per subdimension.

2. **Level 2 (`cascade_level2`)** — subdimension MRP scores are discretised and grouped into macro-dimensions. A second POSet is built at this level and final MRP scores are derived.

This produces the `indicators_macro_dim` dataset variants used in notebooks 050–090.

---

### Minimum Dominating Set

The **Minimum Dominating Set (MDS)** of a POSet is the smallest subset $D \subseteq P$ such that every element is either in $D$ or is dominated by at least one element of $D$:

$$D \subseteq P : \forall\, p \in P,\; p \in D \;\lor\; \exists\, d \in D : d \succ p$$

In the context of this project, each country is a poset element and country $A$ dominates $B$ when $A$ performs at least as well on every indicator. The MDS corresponds to the set of **maximal elements** — countries that no other country consistently outperforms.

The size and composition of the MDS is a structural descriptor of the poset:
- **Small MDS** (1–2 countries) → clear dominance hierarchy, low incomparability.
- **Large MDS** → high structural incomparability, no single country excels across all dimensions.

Notebook 070 reports and visualises the MDS for each dataset and year.

---

### Posetic Separation

The **Posetic separation** $\text{Sep}(i,j)$ measures how far apart two countries rank on average across all possible linear extensions of the poset:

$$\text{Sep}(i,j) = \frac{\mathbb{E}_{\sigma \sim \text{LE}(P)}\bigl[|\sigma(i) - \sigma(j)|\bigr]}{n - 1} \in [0,1]$$

A value near 1 means $i$ and $j$ consistently occupy opposite ends of the ranking; a value near 0 means they rank similarly in most extensions. The expectation is approximated via **Bubley-Dyer MCMC** sampling (50 000 linear extensions per poset).

Separation is used in two ways in this project:
- **Notebook 080** — as a per-poset matrix showing the pairwise structural distance between countries.
- **Notebook 090** — as a dissimilarity input to Classical MDS, enabling a geometric embedding of countries.

### MDS Country Projection

Countries are embedded in 2D/3D space via **Classical Multi-Dimensional Scaling (cMDS / PCoA)** using the posetic separation matrix as input. Embeddings for 2015 and 2018 are independently computed and then aligned with **Procrustes analysis** (rotation + reflection, no scaling) so that arrows between the same country across years represent genuine positional shifts in the dominance space.

---

## Project Structure

### Analysis Pipeline

| Notebook | Description |
|---|---|
| [`010_data_extractor.ipynb`](notebooks/010_data_extractor.ipynb) | Data extraction from the OECD API → `data/010_child_well_being.parquet` |
| [`020_data_exploration.ipynb`](notebooks/020_data_exploration.ipynb) | Exploratory data analysis and correlation study |
| [`030_data_transformer.ipynb`](notebooks/030_data_transformer.ipynb) | Normalization and discretization of indicators |
| [`040_data_partitioner.ipynb`](notebooks/040_data_partitioner.ipynb) | Partition by year (2015 / 2018) and domain; produce all `040_*.parquet` files |
| [`050_Poset_creator.ipynb`](notebooks/050_Poset_creator.ipynb) | Build POSets from all partitions → `data/050_posets*.pkl` |
| [`054_Poset_check.Rmd`](notebooks/054_Poset_check.Rmd) | R reference implementation (poseticDataAnalysis) for cross-language validation |
| [`055_Poset_check.ipynb`](notebooks/055_Poset_check.ipynb) | Python-side cross-language validation against the R reference |
| [`060_Poset_analysis.ipynb`](notebooks/060_Poset_analysis.ipynb) | Analysis of POSet structures: minimals, maximals, MDS width, separation |
| [`070_MDS_visualization.ipynb`](notebooks/070_MDS_visualization.ipynb) | Minimum Dominating Set analysis and visualisation for each POSet |
| [`080_dominance_matrices.ipynb`](notebooks/080_dominance_matrices.ipynb) | Certain / possible dominance heatmaps and posetic separation matrices per dataset |
| [`090_MDS_projection.ipynb`](notebooks/090_MDS_projection.ipynb) | cMDS + Procrustes projection using posetic separation: country trajectories 2015 → 2018 |

### Discretization Variants

| Dataset | Levels | Domains |
|---|---|---|
| `indicators_macro_dim` | 3, 4, 5 | Child outcome macro-dimensions |
| `indicators_dim_discrete` | 3, 4 | Individual indicators |
| `public_expenditure_dim_discrete` | 3, 4 | Public expenditure by category |

### `scripts/` Module

| Module | Content |
|---|---|
| `transformer.py` | `normalize_minmax` — directed min-max normalisation; `discretize` — ordinal discretisation (quantile or equal-width); `cascade_aggregate` — Level 1 MRP cascade (indicators → subdimensions); `cascade_level2` — Level 2 MRP cascade (subdimensions → macro-dimensions) |

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

The library in `poset/` requires no installation — notebooks reference it via `sys.path.insert(0, '..')`.

---

## Analysed Metrics

Metrics follow the OECD CWB framework: **A** (child outcomes), **B** (child circumstances), **C** (policies for children).

---

### A — Child Outcomes

#### A1 — Material Conditions

**A1_2 · Children experiencing severe housing deprivation**
> 0- to 17-year-olds living in households experiencing severe housing deprivation (Eurostat definition). Severe housing deprivation is defined and measured in line with the Eurostat definition. Under the Eurostat definition, a household experiencing severe housing deprivation is one that is both overcrowded and experiencing one or more of the following: The dwelling has a leaking roof, damp walls, floors or foundation, or rot in window frames or floor; The dwelling has neither a bath nor a shower; The dwelling has no flushing toilet for exclusive use of the household; The dwelling is considered too dark.

**A1_4 · Students who report not having an internet connection at home**
> 15-year-old students who report not having an internet connection at home. 15-year-old students were asked 'Which of the following are in your home? ... A link to the Internet' and presented with the response options 'Yes' and 'No'. Data refer to the percent responding 'No'. '15-year-olds' and/or '15-year-old students' are used as shorthand for the PISA target population.

#### A2 — Health

**A2_1 · Infant mortality rates**
> Deaths of children aged less than one (no minimum threshold of gestation period or birthweight). Infant mortality is defined as deaths of children aged less than one year (no minimum threshold of gestation period or birthweight).

#### A3 — Education

**A3_3 · Top performers in reading, maths and/or science**
> 15-year-old students who attained Level 5 or 6 in at least one of the three main PISA test subjects (reading, mathematics and science). Data refer to the percent of 15-year-old students who attained Level 5 or 6 in at least one of the three main PISA test subjects (reading, mathematics and science). For more detail on the construction of the PISA proficiency scales and proficiency levels, see the PISA 2018 Technical Report and the corresponding Technical Reports from earlier rounds.

**A3_4 · Students who expect to complete tertiary education**
> 15-year-old students who report expecting to complete tertiary education. Data refer to the percent responding either '(ISCED level 5B)' or '(ISCED level 5A or 6)'. Percent among valid responses only.

**A3_5 · Youth not in education, employment or training (NEET)**
> 15- to 29-year-olds not in education, employment or training (NEET). Children and young people are classified as 'NEET' if they had neither received formal education and/or training in the regular educational system in the four weeks prior to being surveyed, nor were either working for pay or profit for at least one hour or had a job but were temporarily not at work during the survey reference week.

#### A4 — Subjective Well-being

**A4_6 · Students who report high satisfaction with their life as a whole**
> 15-year-old students who report high satisfaction with their life as a whole. Data are based on responses by students to the question 'Overall, how satisfied are you with your life as a whole these days?'. Students were asked to record their response on a scale from 0 to 10. Students recording a 9 or a 10 were classified as reporting high satisfaction.

---

### B — Child Circumstances

#### B1 — Economic Circumstances

**B1_1 · Children living in relative income poverty**
> 0- to 17-year-olds in relative income poverty. Data are based on equivalised household disposable income. The poverty threshold is set at 50% of median disposable income in each country.

**B1_5 · Students who firmly report that their parents encourage them to be confident**
> 15-year-old students who strongly agree with the statement 'My parents encourage me to be confident'. Data refer to the percent responding 'Strongly agree'. Percent among valid responses only.

#### B2 — Social Circumstances

**B2_1 · Children enrolled in early childhood education and care**
> 0- to 2-year-olds participating in formal early childhood education and care services. Data generally include children enrolled in early childhood education services (ISCED 2011 level 0) and other registered ECEC services.

**B2_4 · Students who report experiencing bullying at school**
> 15-year-old students who report experiencing any of a specified list of bullying acts at school at least a few times a month. Data refer to the percent of 15-year-old students who responded to at least one bullying behaviour with 'a few times a month' or 'once a week or more'.

**B2_5 · Students who feel like they belong at school**
> 15-year-old students who agree (or strongly agree) with the statement 'I feel like I belong at school'. Data refer to the percent responding 'Agree' or 'Strongly Agree'. Percent among valid responses only.

#### B3 — Safety and Security

**B3_5 · Children in households that report crime and violence in their local area**
> 0- to 17-year-olds in households that report problems with crime or violence in the area. Data refer to the percent of children in households that report problems with crime, violence or vandalism in the area in which they live.

#### B4 — Digital Environment

**B4_3 · Students who firmly believe the Internet is a great resource for information**
> 15-year-old students who 'strongly agree' with the statement 'The Internet is a great resource for obtaining information I am interested in'. Data refer to the percent responding 'Strongly agree'. Percent among valid responses only.

---

### C — Policies for Children

#### C1 — Family Support and Income Redistribution

**C1_1 · Public expenditure on families per child**
> Public expenditure on benefits and services exclusively for families and children.

**C1_2 · Difference between before- and after-tax and transfer child relative income poverty rates**
> Percentage point difference between the child relative income poverty rate before and after taxes and transfers.

**C1_3 · Guaranteed minimum income for a jobless couple with two children**
> Modelled disposable income of a jobless couple family with two children (age 4 and 6) claiming Guaranteed Minimum Income benefits, as a percentage of median disposable income.

**C1_4 · Total length of paid maternity and parental leave available to mothers**
> Total length of paid maternity leave and paid parental leave available to mothers after the birth of a child.

**C1_5 · Total length of paid paternity and parental leave reserved for fathers**
> Paid leave reserved for fathers: paternity leave, 'father quotas', and non-transferable periods of paid parental leave.

#### C2 — Housing, Community and Recreation

**C2_1 · General government expenditure on housing and community amenities per person**
> Total general government expenditure classified under COFOG function 06 ('Housing and community amenities').

**C2_2 · General government expenditure on recreation, culture and religion per person**
> Total general government expenditure classified under COFOG function 08 ('Recreation, culture and religion').

**C2_3 · General government expenditure on housing social protection per person**
> Total general government expenditure classified under COFOG function 10.6 ('Social Protection: Housing').

#### C3 — Health

**C3_1 · Government and compulsory contributory health insurance expenditure on health per person**
> Total government and/or compulsory health insurance spending on health (all functions and providers).

**C3_2 · Children less than one year old who have received three doses of the DPT vaccine**
> Percentage of children under one year old who have received three doses of the combined diphtheria-tetanus-pertussis vaccine.

**C3_3 · Children less than one year old who have received at least one dose of measles-containing vaccine**
> Percentage of children under one year old who have received at least one dose of measles-containing vaccine in a given year.

#### C4 — Education and Care

**C4_1 · Public expenditure on early childhood education and care per child**
> Public spending towards formal day-care services and pre-primary education services.

**C4_2 · Typical net childcare costs for parents using centre-based care**
> Net cost of full-time care in a typical childcare centre for a two-child couple family, both parents in full-time employment on low earnings (67% of national average).

**C4_3 · Students to teaching staff ratio in pre-primary education**
> Full-time equivalent number of children enrolled in pre-primary education per full-time equivalent teaching staff member.

**C4_4 · Public expenditure on primary, secondary and post-secondary non-tertiary education per FTE student**
> Final general government expenditure on education institutions (public and private).

**C4_5 · Public expenditure on ancillary education services per FTE student**
> Final general government expenditure on ancillary services (meals, school health, transportation) in primary, secondary and post-secondary non-tertiary education.

**C4_6 · Students to teaching staff ratio in secondary education**
> Full-time equivalent number of students enrolled in secondary education per full-time equivalent teaching staff member.

#### C5 — Environment

**C5_1 · General government expenditure on environment protection per person**
> Total general government expenditure classified under COFOG function 05 ('Environment protection'), covering waste management, waste water management, pollution abatement, and biodiversity protection.
