import pandas as pd
import numpy as np
from drain.util import execute_sql


def follow(id, edges, weak=True, _visited=None):
    """
    Follow the a graph to find the nodes connected to a given node.
    Args:
        id: the id of the starting node
        edges: a pandas DataFrame of edges. Each row is an edge with two columns containing the ids of the vertices.
        weak: If True, edges are undirected.
              Otherwise edges are directed from the first column to the second column.
        _visited: used internally for recursion
    Returns: the set of all nodes connected to the starting node.

    """
    if _visited is None:
        _visited = set()
    _visited.add(id)

    for row in edges[edges.ix[:,0] == id].values:
        if(row[1] not in _visited):
            follow(row[1], edges, weak, _visited)

    if weak:
        for row in edges[edges.ix[:,1] == id].values:
            if(row[0] not in _visited):
                follow(row[0], edges, weak, _visited)

    return _visited


def get_components(edges, vertices=None):
    """
    if sparse (a lot of disconnected vertices) find those separately (faster)
    """
    if vertices is None:
        vertices = pd.DataFrame({'id': pd.concat((edges['id1'], edges['id2'])).unique()})

    visited = set()
    components = {}

    for id1 in vertices.values[:, 0]:
        if id1 not in visited:
            c = follow(id1, edges)
            visited.update(c)
            components[id1] = c

    return components


def components_dict_to_df(components):
    deduped = np.empty((0, 2), dtype=int)

    for id1 in components:
        deduped = np.append(deduped, [[id1, id2] for id2 in components[id1]], axis=0)

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
