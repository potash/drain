import pandas as pd
import numpy as np
from itertools import chain

from drain.util import execute_sql


def follow(id, edges, directed=False, _visited=None):
    """
    Follow the a graph to find the nodes connected to a given node.
    Args:
        id: the id of the starting node
        edges: a pandas DataFrame of edges. Each row is an edge with two columns containing
            the ids of the vertices.
        directed: If True, edges are directed from first column to second column.
              Otherwise edges are undirected.
        _visited: used internally for recursion
    Returns: the set of all nodes connected to the starting node.

    """
    if _visited is None:
        _visited = set()
    _visited.add(id)

    for row in edges[edges.ix[:, 0] == id].values:
        if(row[1] not in _visited):
            follow(row[1], edges, directed, _visited)

    if not directed:
        for row in edges[edges.ix[:, 1] == id].values:
            if(row[0] not in _visited):
                follow(row[0], edges, directed, _visited)

    return _visited


def get_components(edges, vertices=None):
    """
    Return connected components from graph determined by edges matrix
    Args:
        edges: DataFrame of (undirected) edges.
        vertices: set of vertices in graph. Defaults to union of all vertices in edges.

    Returns:
        set of connected components, each of which is a set of vertices.

    """
    if vertices is None:
        vertices = set(chain(edges.ix[:, 0], edges.ix[:, 1]))

    visited = set()
    components = []

    for id in vertices:
        if id not in visited:
            c = follow(id, edges)
            visited.update(c)
            components.append(c)

    return components


def components_to_df(components, id_func=None):
    """
    Convert components to a join table with columns id1, id2
    Args:
        components: A collection of components, each of which is a set of vertex ids.
            If a dictionary, then the key is the id for the component. Otherwise,
            the component id is determined by applying id_func to the component.
        id_func: If components is a dictionary, this should be None. Otherwise,
            this is a callable that, given a set of vertices, deermines the id.
            If components is not a dict and id_func is None, it defaults to `min`.
    Returns: A dataframe representing the one-to-many relationship between
            component names (id1) and their members (id2).
    """
    deduped = np.empty((0, 2), dtype=int)

    if id_func is None:
        if isinstance(components, dict):
            raise ValueError("If components is a dict, id_func should be None.")
        else:
            id_func = min

    for c in components:
        if id_func is None:
            id1 = c
            c = components[c]
        else:
            id1 = id_func(c)

        deduped = np.append(deduped, [[id1, id2] for id2 in c], axis=0)

    deduped = pd.DataFrame(deduped, columns=['id1', 'id2'])
    return deduped


def insert_singletons(source_table, dest_table, id_column, engine):
    sql = """
    WITH singletons as (
        select distinct {id_column} id from {source_table}
        left join {dest_table} on {source_table}.{id_column} = {dest_table}.id2
        where {dest_table}.id2 is null
    )

    INSERT INTO {dest_table}
        SELECT id,id from singletons;
    """.format(source_table=source_table, dest_table=dest_table, id_column=id_column)

    execute_sql(sql, engine)
