import itertools
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import chain

from pgmpy.factors import factor_product
from pgmpy.models import BayesianModel, JunctionTree
from pgmpy.inference.EliminationOrder import (
    WeightedMinFill,
    MinNeighbors,
    MinFill,
    MinWeight,
)
from pgmpy.factors.discrete import TabularCPD


class VariableElimination(object):
    def __init__(self, model):
        self.model = model
        model.check_model()

        if isinstance(model, JunctionTree):
            self.variables = set(chain(*model.nodes()))
        else:
            self.variables = model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        if isinstance(model, BayesianModel):
            self.state_names_map = {}
            for node in model.nodes():
                cpd = model.get_cpds(node)
                if isinstance(cpd, TabularCPD):
                    self.cardinality[node] = cpd.variable_card
                    cpd = cpd.to_factor()
                for var in cpd.scope():
                    self.factors[var].append(cpd)
                self.state_names_map.update(cpd.no_to_name)

        elif isinstance(model, JunctionTree):
            self.cardinality = model.get_cardinality()

            for factor in model.get_factors():
                for var in factor.variables:
                    self.factors[var].append(factor)

    def _get_working_factors(self, evidence):
        """
        Uses the evidence given to the query methods to modify the factors before running
        the variable elimination algorithm.
        Parameters
        ----------
        evidence: dict
            Dict of the form {variable: state}
        Returns
        -------
        dict: Modified working factors.
        """

        working_factors = {
            node: {(factor, None) for factor in self.factors[node]}
            for node in self.factors
        }

        # Dealing with evidence. Reducing factors over it before VE is run.
        if evidence:
            for evidence_var in evidence:
                for factor, origin in working_factors[evidence_var]:
                    factor_reduced = factor.reduce(
                        [(evidence_var, evidence[evidence_var])], inplace=False
                    )
                    for var in factor_reduced.scope():
                        working_factors[var].remove((factor, origin))
                        working_factors[var].add((factor_reduced, evidence_var))
                del working_factors[evidence_var]
        return working_factors

    def _get_elimination_order(
        self, variables, evidence, elimination_order, show_progress=True
    ):
        """
        Deals with all elimination order parameters given to _variable_elimination method
        and returns a list of variables that are to be eliminated
        Parameters
        ----------
        elimination_order: str or list
        Returns
        -------
        list: A list of variables names in the order they need to be eliminated.
        """
        to_eliminate = (
            set(self.variables)
            - set(variables)
            - set(evidence.keys() if evidence else [])
        )

        # Step 1: If elimination_order is a list, verify it's correct and return.
        if hasattr(elimination_order, "__iter__") and (
            not isinstance(elimination_order, str)
        ):
            if any(
                var in elimination_order
                for var in set(variables).union(
                    set(evidence.keys() if evidence else [])
                )
            ):
                raise ValueError(
                    "Elimination order contains variables which are in"
                    " variables or evidence args"
                )
            else:
                return elimination_order

        # Step 2: If elimination order is None or a Markov model, return a random order.
        elif (elimination_order is None) or (not isinstance(self.model, BayesianModel)):
            return to_eliminate

        # Step 3: If elimination order is a str, compute the order using the specified heuristic.
        elif isinstance(elimination_order, str) and isinstance(
            self.model, BayesianModel
        ):
            heuristic_dict = {
                "weightedminfill": WeightedMinFill,
                "minneighbors": MinNeighbors,
                "minweight": MinWeight,
                "minfill": MinFill,
            }
            elimination_order = heuristic_dict[elimination_order.lower()](
                self.model
            ).get_elimination_order(nodes=to_eliminate, show_progress=show_progress)
            return elimination_order

    def _variable_elimination(
        self,
        variables,
        operation,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=True,
    ):
        """
        Implementation of a generalized variable elimination.
        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.
        operation: str ('marginalize' | 'maximize')
            The operation to do for eliminating the variable.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: str or list (array-like)
            If str: Heuristic to use to find the elimination order.
            If array-like: The elimination order to use.
            If None: A random elimination order is used.
        """
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables is not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            if joint:
                return factor_product(*set(all_factors))
            else:
                return set(all_factors)

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        working_factors = self._get_working_factors(evidence)
        elimination_order = self._get_elimination_order(
            variables, evidence, elimination_order, show_progress=show_progress
        )

        # Step 3: Run variable elimination
        if show_progress:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        for var in pbar:
            if show_progress:
                pbar.set_description(f"Eliminating: {var}")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = set()
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add((factor, origin))
        final_distribution = [factor for factor, _ in final_distribution]

        if joint:
            if isinstance(self.model, BayesianModel):
                return factor_product(*final_distribution).normalize(inplace=False)
            else:
                return factor_product(*final_distribution)
        else:
            query_var_factor = {}
            for query_var in variables:
                phi = factor_product(*final_distribution)
                query_var_factor[query_var] = phi.marginalize(
                    list(set(variables) - set([query_var])), inplace=False
                ).normalize(inplace=False)
            return query_var_factor

    def query(
        self,
        variables,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=False,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically
        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.
        """
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        return self._variable_elimination(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            elimination_order=elimination_order,
            joint=joint,
            show_progress=show_progress,
        )

    def max_marginal(
        self,
        variables=None,
        evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Computes the max-marginal over the variables given the evidence.
        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically
        """
        if not variables:
            variables = []

        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables if variables is not None else [])
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        final_distribution = self._variable_elimination(
            variables=variables,
            operation="maximize",
            evidence=evidence,
            elimination_order=elimination_order,
            show_progress=show_progress,
        )

        return np.max(final_distribution.values)

    def map_query(
        self,
        variables=None,
        evidence=None,
        elimination_order="MinFill",
        show_progress=False,
    ):
        """
        Computes the MAP Query over the variables given the evidence.
        Note: When multiple variables are passed, it returns the map_query for each
        of them individually.
        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically
        """
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables if variables is not None else [])
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        final_distribution = self._variable_elimination(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            elimination_order=elimination_order,
            joint=True,
            show_progress=show_progress,
        )
        argmax = np.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        if not variables:
            return map_query_results
        else:
            return_dict = {}
            for var in variables:
                return_dict[var] = map_query_results[var]
            return return_dict

    def induced_graph(self, elimination_order):
        """
        Returns the induced graph formed by running Variable Elimination on the network.
        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.
        """
        # If the elimination order does not contain the same variables as the model
        if set(elimination_order) != set(self.variables):
            raise ValueError(
                "Set of variables in elimination order"
                " different from variables in model"
            )

        eliminated_variables = set()
        working_factors = {
            node: [factor.scope() for factor in self.factors[node]]
            for node in self.factors
        }

        # The set of cliques that should be in the induced graph
        cliques = set()
        for factors in working_factors.values():
            for factor in factors:
                cliques.add(tuple(factor))

        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        for var in elimination_order:
            factors = [
                factor
                for factor in working_factors[var]
                if not set(factor).intersection(eliminated_variables)
            ]
            phi = set(itertools.chain(*factors)).difference({var})
            cliques.add(tuple(phi))
            del working_factors[var]
            for variable in phi:
                working_factors[variable].append(list(phi))
            eliminated_variables.add(var)

        edges_comb = [
            itertools.combinations(c, 2) for c in filter(lambda x: len(x) > 1, cliques)
        ]
        return nx.Graph(itertools.chain(*edges_comb))

    def induced_width(self, elimination_order):
        """
        Returns the width (integer) of the induced graph formed by running Variable Elimination on the network.
        The width is the defined as the number of nodes in the largest clique in the graph minus 1.
        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.
        """
        induced_graph = self.induced_graph(elimination_order)
        return nx.graph_clique_number(induced_graph) - 1

