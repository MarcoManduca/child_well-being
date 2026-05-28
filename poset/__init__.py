"""
poseticDataAnalysis - Python port
==================================
Partially Ordered Set (POSet) analysis library.
Ported from the R package poseticDataAnalysis by Avellone, De Capitani, Fattore.

Added some implementations to work with Polars dataframes and handle `null` values as uncertainty intervals in the hyperlattice.
Authors: Simone Caglio supported by Claude Code (Python port),
Original authors of poseticDataAnalysis R package: Michele Fattore, Luca De Capitani, Andrea Avellone, Andrea Suardi.

Reference:
    Fattore M., De Capitani L., Avellone A., Suardi A. (2024).
    A fuzzy posetic toolbox for multi-criteria evaluation on ordinal data systems.
    Annals of Operations Research. doi:10.1007/s10479-024-06352-3
"""

from .poset import POSet, LinearPOSet, BinaryVariablePOSet
from .poset_ops import (
    ProductPOSet, DisjointSumPOSet, LinearSumPOSet,
    DualPOSet, IntersectionPOSet, LiftingPOSet,
    LexicographicProductPOSet, CrownPOSet, FencePOSet,
)
from .relations import (
    IsReflexive, IsSymmetric, IsAntisymmetric, IsTransitive,
    IsPartialOrder, IsPreorder,
    TransitiveClosure, ReflexiveClosure,
)
from .poset_query import (
    POSetElements,
    DominanceMatrix, Dominates, IsDominatedBy,
    IsComparableWith, IsIncomparableWith,
    ComparabilitySetOf, IncomparabilitySetOf,
    IncomparabilityRelation, OrderRelation,
    UpsetOf, DownsetOf, IsUpset, IsDownset,
    POSetMaximals, POSetMinimals,
    IsMaximal, IsMinimal,
    CoverRelation, CoverMatrix,
    POSetMeet, POSetJoin,
    IsExtensionOf,
)
from .linear_extensions import (
    LEGenerator, LEBubleyDyer, LEGet,
)
from .mrp import (
    ExactMRP, BubleyDyerMRPGenerator, BubleyDyerMRP,
    LexMRP,
)
from .separation import (
    ExactSeparation, BubleyDyerSeparation,
    BuildBubleyDyerSeparationGenerator,
    LexSeparation,
)
from .evaluation import (
    ExactEvaluation,
    BuildBubleyDyerEvaluationGenerator, BubleyDyerEvaluation,
)
from .dominance import BLSDominance
from .fuzzy import (
    FuzzyInBetweenness, FuzzyInBetweennessMinMax, FuzzyInBetweennessProbabilistic,
    FuzzySeparation, FuzzySeparationMinMax, FuzzySeparationProbabilistic,
)
from .embedding import (
    BidimensionalPosetRepresentation,
    OptimalBidimensionalEmbedding,
)
from .from_polars import poset_from_polars, interval_summary

__version__ = "1.0.0"
__author__ = "Python port of poseticDataAnalysis (Avellone, De Capitani, Fattore)"
