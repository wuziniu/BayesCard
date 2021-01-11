import itertools
import networkx as nx
import numpy as np
import time
try:
    from tqdm import tqdm
except:
    import tqdm
from collections import defaultdict
from itertools import chain
import copy
from Pgmpy.inference import Inference
from Pgmpy.factors import factor_product
from Pgmpy.models import BayesianModel, JunctionTree
from Pgmpy.inference.EliminationOrder import (
    WeightedMinFill,
    MinNeighbors,
    MinFill,
    MinWeight,
)
from Pgmpy.factors.discrete import TabularCPD


class VariableEliminationJIT(object):
    def __init__(self, model, cpds, topological_order, topological_order_node, fanouts=None, probs=None, root=True):
        model.check_model()
        self.cpds = cpds
        self.topological_order = topological_order
        self.topological_order_node = topological_order_node
        self.model = model
        self.fanouts = fanouts
        if probs is not None:
            self.probs = probs
        else:
            self.probs = dict()

        self.variables = model.nodes()

        if root:
            self.root = self.get_root()

    def get_root(self):
        """Returns the network's root node."""

        def find_root(graph, node):
            predecessor = next(self.model.predecessors(node), None)
            if predecessor:
                root = find_root(graph, predecessor)
            else:
                root = node
            return root

        return find_root(self, list(self.model.nodes)[0])

    def steiner_tree(self, nodes):
        """Returns the minimal part of the tree that contains a set of nodes."""
        sub_nodes = set()

        def walk(node, path):
            if len(nodes) == 0:
                return

            if node in nodes:
                sub_nodes.update(path + [node])
                nodes.remove(node)

            for child in self.model.successors(node):
                walk(child, path + [node])

        walk(self.root, [])
        sub_graph = self.model.subgraph(sub_nodes)
        sub_graph.cardinalities = defaultdict(int)
        for node in sub_graph.nodes:
            sub_graph.cardinalities[node] = self.model.cardinalities[node]
        return sub_graph


    def _get_working_factors(self, query=None, fanout=[]):
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
        useful_var = list(query.keys()) + fanout
        sub_graph_model = self.steiner_tree(useful_var)

        elimination_order = []
        working_cpds = []
        working_factors = dict()
        for i, node in enumerate(self.topological_order_node[::-1]):
            ind = len(self.topological_order_node)-i-1
            if node in sub_graph_model.nodes:
                elimination_order.append(node)
                cpd = copy.deepcopy(self.cpds[ind])
                working_cpds.append(cpd)
                working_factors[node] = [cpd]

        for node in sub_graph_model.nodes:
            for cpd in working_cpds:
                if node != cpd.variable and node in cpd.variables:
                    working_factors[node].append(cpd)

        return working_factors, sub_graph_model, elimination_order


    def query(self, query, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        n_distinct: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        """
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query)
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1)
            #print(var, len(working_factors[var]), root_var)
            if len(working_factors[var]) == 1:
                #leaf node in BN
                if var in query:
                    #print(var, query[var], n_distinct[var])
                    new_value = working_factors[var][0].values
                    if n_distinct:
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                        else:
                            new_value = np.dot(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                    new_value = new_value.reshape(-1)
                    if root_var:
                        return new_value
                    assert len(new_value.shape) == 1, \
                        f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        return 1
                    new_value = np.ones(working_factors[var][0].values.shape[-1])

                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    self_value = working_factors[var][0].values[query[var]]  #Pr(var|Parent(var))
                    if n_distinct:
                        self_value = (self_value.transpose() * n_distinct[var]).transpose()
                    children_value = []
                    #check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        #print("y")
                        #print(cpd.variables)
                        child_value = cpd.values[query[var]]    #M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                        f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        #print(children_value)
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)
                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                        f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
        return 0

    def expectation(self, query, fanout_attrs, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the expectation inference
        """
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, fanout_attrs)
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1)
            #print(var, len(working_factors[var]))
            if len(working_factors[var]) == 1:
                #leaf node in BN
                if var in query:
                    new_value = working_factors[var][0].values
                    #print(new_value.shape)
                    if n_distinct:
                        #print(var, query[var], n_distinct[var])
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                            new_value = new_value.reshape(-1)
                        else:
                            new_value = np.dot(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                    new_value = new_value.reshape(-1)
                    if root_var:
                        return new_value
                elif var in fanout_attrs:
                    assert not root_var, "no querying variables"
                    new_value = working_factors[var][0].values
                    #print(new_value.shape)
                    new_value = np.dot(self.fanouts[var], new_value)
                else:
                    if root_var:
                        return 1
                    new_value = np.ones(working_factors[var][0].values.shape[-1])
                    
                #print(new_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var} with shape {new_value.shape}"
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    self_value = working_factors[var][0].values[query[var]]  #Pr(var|Parent(var))
                    if n_distinct:
                        self_value = (self_value.transpose() * n_distinct[var]).transpose()
                    children_value = []
                    #check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        #print("y")
                        #print(cpd.variables)
                        child_value = cpd.values[query[var]]    #M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                        f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        #print(children_value)
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)

                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    if var in fanout_attrs:
                        self_value = (self_value.transpose() * self.fanouts[var]).transpose()
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        #print(cpd.variables)
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                        f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                #print(new_value)
                working_factors[var][0].values = new_value
        return 0


class VariableElimination(object):
    def __init__(self, model, probs=None, root=True):
        model.check_model()
        self.model = model
        if probs is not None:
            self.probs = probs
        else:
            self.probs = dict()

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
        if root:
            self.root = self.get_root()

    def get_root(self):
        """Returns the network's root node."""

        def find_root(graph, node):
            predecessor = next(self.model.predecessors(node), None)
            if predecessor:
                root = find_root(graph, predecessor)
            else:
                root = node
            return root

        return find_root(self, list(self.model.nodes)[0])

    def steiner_tree(self, nodes):
        """Returns the minimal part of the tree that contains a set of nodes."""
        sub_nodes = set()

        def walk(node, path):
            if len(nodes) == 0:
                return

            if node in nodes:
                sub_nodes.update(path + [node])
                nodes.remove(node)

            for child in self.model.successors(node):
                walk(child, path + [node])

        walk(self.root, [])
        sub_graph = self.model.subgraph(sub_nodes)
        sub_graph.cardinalities = defaultdict(int)
        for node in sub_graph.nodes:
            sub_graph.cardinalities[node] = self.model.cardinalities[node]
        return sub_graph

    def get_probs(self, attribute, values):
        """
        Calculate Pr(attr in values) where values must be a list
        """
        factor = self.probs[attribute]
        values = [factor.get_state_no(attribute, no) for no in values]
        return np.sum(factor.values[values])


    def _get_working_factors(self, variables=[], evidence=None, return_probs=False, reduce=True, BP=False):
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
        if BP:
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
                            if var in working_factors:
                                working_factors[var].remove((factor, origin))
                                working_factors[var].add((factor_reduced, evidence_var))
                    if type(evidence[evidence_var]) != list:
                        del working_factors[evidence_var]
            return working_factors

        else:
            useful_var = copy.deepcopy(variables)
            if evidence:
                useful_var += list(evidence.keys())
            sub_graph_model = self.steiner_tree(useful_var)
            variables_sub_graph = set(sub_graph_model.nodes)

            working_factors = dict()
            for node in sub_graph_model.nodes:
                working_factors[node] = set()
                for factor in self.factors[node]:
                    if set(factor.variables).issubset(variables_sub_graph):
                        working_factors[node].add((factor, None))

            if return_probs:
                probs = dict()
            # Dealing with evidence. Reducing factors over it before VE is run.
            if evidence and reduce:
                for evidence_var in evidence:
                    for factor, origin in working_factors[evidence_var]:
                        factor_reduced = factor.reduce(
                            [(evidence_var, evidence[evidence_var])], inplace=False
                        )
                        if return_probs:
                            factor_reduced.normalize()
                            probs[evidence_var] = self.get_probs(evidence_var, evidence[evidence_var])
                        for var in factor_reduced.scope():
                            if var in working_factors:
                                working_factors[var].remove((factor, origin))
                                working_factors[var].add((factor_reduced, evidence_var))
                    if type(evidence[evidence_var]) != list:
                        del working_factors[evidence_var]
            if return_probs:
                return working_factors, sub_graph_model, probs
            return working_factors, sub_graph_model

    def _get_elimination_order(
            self, variables=None, evidence=None, model=None, elimination_order="minfill", show_progress=False
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
        if model is None:
            model = self.model
        if isinstance(model, JunctionTree):
            all_variables = set(chain(*model.nodes()))
        else:
            all_variables = model.nodes()

        if variables is None:
            to_eliminate = set(all_variables)
        else:
            not_evidence_eliminate = []
            if evidence is not None:
                for key in evidence:
                    if type(evidence[key]) != list:
                        not_evidence_eliminate.append(key)
            to_eliminate = (
                    set(all_variables)
                    - set(variables)
                    - set(not_evidence_eliminate)
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
        elif (elimination_order is None) or (not isinstance(model, BayesianModel)):
            return to_eliminate

        # Step 3: If elimination order is a str, compute the order using the specified heuristic.
        elif isinstance(elimination_order, str) and isinstance(
                model, BayesianModel
        ):
            heuristic_dict = {
                "weightedminfill": WeightedMinFill,
                "minneighbors": MinNeighbors,
                "minweight": MinWeight,
                "minfill": MinFill,
            }
            elimination_order = heuristic_dict[elimination_order.lower()](
                model
            ).get_elimination_order(nodes=to_eliminate, show_progress=show_progress)
            return elimination_order

    def _variable_elimination(
            self,
            variables,
            operation,
            evidence=None,
            elimination_order="minfill",
            joint=True,
            show_progress=False,
            BP=False
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
        # tic = time.time()
        if BP:
            working_factors = self._get_working_factors(variables, evidence, BP=BP)
            elimination_order = self._get_elimination_order(
                variables, evidence, elimination_order=elimination_order, show_progress=show_progress
            )
        else:
            working_factors, sub_graph_model = self._get_working_factors(variables, evidence)
            elimination_order = self._get_elimination_order(
                variables, evidence, sub_graph_model, elimination_order, show_progress=show_progress
            )
        # print(f"getting elimination orders takes {time.time()-toc} secs")
        # Step 3: Run variable elimination
        if show_progress:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        for var in pbar:
            #tic = time.time()
            # print(var)
            if show_progress:
                pbar.set_description("Eliminating: {var}".format(var=var))
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
                if variable in working_factors:
                    working_factors[variable].add((phi, var))
            eliminated_variables.add(var)
            # print(f"eliminating {var} takes {time.time()-tic} secs")

        # Step 4: Prepare variables to be returned.
        #tic = time.time()
        final_distribution = set()
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add((factor, origin))
        final_distribution = [factor for factor, _ in final_distribution]
        # print(final_distribution)
        # print(f"the rest takes {time.time()-tic} secs")
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
            elimination_order="weightedminfill",
            joint=True,
            show_progress=False,
            BP=False
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
            BP=BP
        )


    def probabilistic_query(
            self,
            query,
            elimination_order="minfill",
            show_progress=False,
    ):
        """
        An efficient implementation of probabilistic query
        But in practice is not as efficient as the previous one, I'm so currently working it.
        Parameters
        ----------
        query: Q of form {"attr_name": attr_values}
               attr_values must be a list of values.
        elimination_order: str or list (array-like)
            If str: Heuristic to use to find the elimination order.
            If array-like: The elimination order to use.
            If None: A random elimination order is used.
        """

        # Step 1: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        # tic = time.time()
        working_factors, sub_graph_model = self._get_working_factors(evidence=query, reduce=False)
        # toc = time.time()
        # print(f"getting working factors takes {toc-tic} secs")

        elimination_order_rest = self._get_elimination_order(variables=list(query.keys()),
                                                        model=sub_graph_model,
                                                        elimination_order=elimination_order,
                                                        show_progress=show_progress)

        elimination_order_variable = self._get_elimination_order(variables=list(elimination_order_rest),
                                                             model=sub_graph_model,
                                                             elimination_order=elimination_order,
                                                             show_progress=show_progress)

        elimination_order = elimination_order_rest + elimination_order_variable
        # print(f"getting elimination orders takes {time.time()-toc} secs")
        # Step 3: Run variable elimination
        if show_progress:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        result = None
        for var in pbar:
            #tic = time.time()
            # print(var)
            if show_progress:
                pbar.set_description("Eliminating: {var}".format(var=var))
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factor_product(*factors)
            #If its a query variable, we record its probability
            if var in query:
                if result is None:
                    result = self.get_probs(var, query[var])
                else:
                    marg_var = [attr for attr in list(phi.variables) if attr != var]
                    phi_var = phi.marginalize(marg_var, inplace=False)
                    phi_var.normalize()
                    values = [phi_var.get_state_no(var, no) for no in query[var]]
                    result *= np.sum(phi_var.values[values])
                for factor, origin in working_factors[var]:
                    factor_reduced = factor.reduce(
                        [(var, query[var])], inplace=False
                    )
                    for fact_var in factor_reduced.scope():
                        if fact_var in working_factors:
                            working_factors[fact_var].remove((factor, origin))
                            working_factors[fact_var].add((factor_reduced, var))
            phi = phi.marginalize([var], inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                if variable in working_factors:
                    working_factors[variable].add((phi, var))

            eliminated_variables.add(var)
            # print(f"eliminating {var} takes {time.time()-tic} secs")

        return result


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
        show_progress=True,
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


class BeliefPropagation(Inference):
    """
    Class for performing inference using Belief Propagation method.

    Creates a Junction Tree or Clique Tree (JunctionTree class) for the input
    probabilistic graphical model and performs calibration of the junction tree
    so formed using belief propagation.

    Parameters
    ----------
    model: BayesianModel, MarkovModel, FactorGraph, JunctionTree
        model for which inference is to performed
    """

    def __init__(self, model):
        super(BeliefPropagation, self).__init__(model)

        if not isinstance(model, JunctionTree):
            self.junction_tree = model.to_junction_tree()
        else:
            self.junction_tree = copy.deepcopy(model)

        self.clique_beliefs = {}
        self.sepset_beliefs = {}

    def get_cliques(self):
        """
        Returns cliques used for belief propagation.
        """
        return self.junction_tree.nodes()

    def get_clique_beliefs(self):
        """
        Returns clique beliefs. Should be called after the clique tree (or
        junction tree) is calibrated.
        """
        return self.clique_beliefs

    def get_sepset_beliefs(self):
        """
        Returns sepset beliefs. Should be called after clique tree (or junction
        tree) is calibrated.
        """
        return self.sepset_beliefs

    def _update_beliefs(self, sending_clique, recieving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message
        recieving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node receiving the message
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        sepset = frozenset(sending_clique).intersection(frozenset(recieving_clique))
        sepset_key = frozenset((sending_clique, recieving_clique))

        # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
        # marginalize the clique over the sepset
        sigma = getattr(self.clique_beliefs[sending_clique], operation)(
            list(frozenset(sending_clique) - sepset), inplace=False
        )

        # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
        self.clique_beliefs[recieving_clique] *= (
            sigma / self.sepset_beliefs[sepset_key]
            if self.sepset_beliefs[sepset_key]
            else sigma
        )

        # \mu_{i, j} = \sigma_{i \rightarrow j}
        self.sepset_beliefs[sepset_key] = sigma

    def _is_converged(self, operation):
        """
        Checks whether the calibration has converged or not. At convergence
        the sepset belief would be precisely the sepset marginal.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
            if operation == marginalize, it checks whether the junction tree is calibrated or not
            else if operation == maximize, it checks whether the junction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisfied for

        .. math:: \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}

        and at max calibration this condition would be satisfied

        .. math:: \max_{C_i - S_{i, j}} \beta_i = \max_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
        """
        # If no clique belief, then the clique tree is not calibrated
        if not self.clique_beliefs:
            return False

        for edge in self.junction_tree.edges():
            sepset = frozenset(edge[0]).intersection(frozenset(edge[1]))
            sepset_key = frozenset(edge)
            if (
                edge[0] not in self.clique_beliefs
                or edge[1] not in self.clique_beliefs
                or sepset_key not in self.sepset_beliefs
            ):
                return False

            marginal_1 = getattr(self.clique_beliefs[edge[0]], operation)(
                list(frozenset(edge[0]) - sepset), inplace=False
            )
            marginal_2 = getattr(self.clique_beliefs[edge[1]], operation)(
                list(frozenset(edge[1]) - sepset), inplace=False
            )
            if (
                marginal_1 != marginal_2
                or marginal_1 != self.sepset_beliefs[sepset_key]
            ):
                return False
        return True

    def _calibrate_junction_tree(self, operation):
        """
        Generalized calibration of junction tree or clique using belief propagation. This method can be used for both
        calibrating as well as max-calibrating.
        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {
            clique: self.junction_tree.get_factors(clique)
            for clique in self.junction_tree.nodes()
        }
        self.sepset_beliefs = {
            frozenset(edge): None for edge in self.junction_tree.edges()
        }

        for clique in self.junction_tree.nodes():
            if not self._is_converged(operation=operation):
                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using nieighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(
                    self.junction_tree, clique
                )
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    self._update_beliefs(edge[0], edge[1], operation=operation)
            else:
                break

    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.
        """
        self._calibrate_junction_tree(operation="marginalize")

    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.
        """
        self._calibrate_junction_tree(operation="maximize")

    def _query(
        self, variables, operation, evidence=None, joint=True, show_progress=False
    ):
        """
        This is a generalized query method that can be used for both query and map query.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        References
        ----------
        Algorithm 10.4 Out-of-clique inference in clique tree
        Probabilistic Graphical Models: Principles and Techniques Daphne Koller and Nir Friedman.
        """

        is_calibrated = self._is_converged(operation=operation)
        # Calibrate the junction tree if not calibrated
        if not is_calibrated:
            self.calibrate()

        if not isinstance(variables, (list, tuple, set)):
            query_variables = [variables]
        else:
            query_variables = list(variables)
        query_variables.extend(evidence.keys() if evidence else [])

        # Find a tree T' such that query_variables are a subset of scope(T')
        nodes_with_query_variables = set()
        for var in query_variables:
            nodes_with_query_variables.update(
                filter(lambda x: var in x, self.junction_tree.nodes())
            )
        subtree_nodes = nodes_with_query_variables

        # Conversion of set to tuple just for indexing
        nodes_with_query_variables = tuple(nodes_with_query_variables)
        # As junction tree is a tree, that means that there would be only path between any two nodes in the tree
        # thus we can just take the path between any two nodes; no matter there order is
        for i in range(len(nodes_with_query_variables) - 1):
            subtree_nodes.update(
                nx.shortest_path(
                    self.junction_tree,
                    nodes_with_query_variables[i],
                    nodes_with_query_variables[i + 1],
                )
            )
        subtree_undirected_graph = self.junction_tree.subgraph(subtree_nodes)
        # Converting subtree into a junction tree
        if len(subtree_nodes) == 1:
            subtree = JunctionTree()
            subtree.add_node(subtree_nodes.pop())
        else:
            subtree = JunctionTree(subtree_undirected_graph.edges())

        # Selecting a node is root node. Root node would be having only one neighbor
        if len(subtree.nodes()) == 1:
            root_node = list(subtree.nodes())[0]
        else:
            root_node = tuple(
                filter(lambda x: len(list(subtree.neighbors(x))) == 1, subtree.nodes())
            )[0]
        clique_potential_list = [self.clique_beliefs[root_node]]

        # For other nodes in the subtree compute the clique potentials as follows
        # As all the nodes are nothing but tuples so simple set(root_node) won't work at it would update the set with'
        # all the elements of the tuple; instead use set([root_node]) as it would include only the tuple not the
        # internal elements within it.
        parent_nodes = set([root_node])
        nodes_traversed = set()
        while parent_nodes:
            parent_node = parent_nodes.pop()
            for child_node in set(subtree.neighbors(parent_node)) - nodes_traversed:
                clique_potential_list.append(
                    self.clique_beliefs[child_node]
                    / self.sepset_beliefs[frozenset([parent_node, child_node])]
                )
                parent_nodes.update([child_node])
            nodes_traversed.update([parent_node])

        # Add factors to the corresponding junction tree
        subtree.add_factors(*clique_potential_list)

        # Sum product variable elimination on the subtree
        variable_elimination = VariableElimination(subtree, root=False)
        if operation == "marginalize":
            return variable_elimination.query(
                variables=variables,
                evidence=evidence,
                joint=joint,
                show_progress=show_progress,
                BP=True
            )
        elif operation == "maximize":
            return variable_elimination.map_query(
                variables=variables, evidence=evidence, show_progress=show_progress
            )

    def query(self, variables, evidence=None, joint=True, show_progress=False):
        """
        Query method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        joint: boolean
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

        result = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            joint=joint,
            show_progress=show_progress,
        )
        if joint:
            return result.normalize(inplace=False)
        else:
            return result

    def map_query(self, variables=None, evidence=None, show_progress=True):
        """
        MAP Query method using belief propagation.

        Note: When multiple variables are passed, it returns the map_query for each
        of them individually.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        """
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables if variables is not None else [])
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        if not variables:
            variables = set(self.variables)

        final_distribution = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            show_progress=show_progress,
        )

        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
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

