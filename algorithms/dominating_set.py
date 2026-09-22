"""
Algorithm 4: Greedy Approximate Minimum Dominating Set.

A Dominating Set D in a graph G = (V, E) is a subset of vertices such that every
vertex in V is either in D or adjacent to a vertex in D.
In social networks, placing monitors or fact-checking bots on dominating set nodes
ensures 100% 1-hop observational coverage over the entire network.
"""

import networkx as nx


def compute_approx_dominating_set(G: nx.Graph) -> dict:
    """
    Computes an approximate Minimum Dominating Set using a Greedy heuristic.
    
    Greedy Selection Strategy:
    At each step, select the node that covers the largest number of currently
    uncovered vertices ({node} + uncovered neighbors) until all vertices are covered.
    
    Returns:
        dict: Dominating set nodes, coverage metrics, and academic explanation.
    """
    if G.number_of_nodes() == 0:
        return {
            'dominating_set': [],
            'selected_count': 0,
            'total_users': 0,
            'coverage_percentage': 0.0,
            'coverage_ratio': 0.0,
            'node_coverage_map': {},
            'explanation': "Graph is empty."
        }
        
    all_nodes = set(G.nodes())
    uncovered = set(all_nodes)
    dominating_set = []
    node_coverage_map = {}
    
    # Precompute closed neighborhood for each node: N[u] = {u} U Neighbors(u)
    closed_neighborhoods = {
        node: set(G.neighbors(node)) | {node}
        for node in all_nodes
    }
    
    while uncovered:
        # Choose node with maximum overlap with remaining uncovered nodes
        best_node = None
        max_cover_count = -1
        
        for node in all_nodes:
            cover_count = len(closed_neighborhoods[node] & uncovered)
            if cover_count > max_cover_count:
                max_cover_count = cover_count
                best_node = node
                
        if best_node is None or max_cover_count == 0:
            # All remaining nodes cannot be covered by existing connections (isolated)
            dominating_set.extend(list(uncovered))
            for iso in uncovered:
                node_coverage_map[iso] = [iso]
            break
            
        newly_covered = closed_neighborhoods[best_node] & uncovered
        dominating_set.append(best_node)
        node_coverage_map[best_node] = list(newly_covered)
        uncovered -= newly_covered
        
    total_users = G.number_of_nodes()
    selected_count = len(dominating_set)
    covered_count = total_users - len(uncovered)
    coverage_percentage = round((covered_count / total_users) * 100, 2) if total_users > 0 else 0.0
    coverage_ratio = round((selected_count / total_users) * 100, 2) if total_users > 0 else 0.0
    
    explanation = (
        f"The Minimum Dominating Set problem is NP-hard. We employ a Greedy Approximation algorithm: "
        f"at each iteration, the algorithm picks the node that covers the maximum number of previously "
        f"unmonitored users. By monitoring just {selected_count} sentinel accounts ({coverage_ratio}% of the network), "
        f"the fact-checking system achieves {coverage_percentage}% direct 1-hop coverage over all {total_users} users."
    )
    
    return {
        'dominating_set': sorted(dominating_set),
        'selected_count': selected_count,
        'total_users': total_users,
        'coverage_percentage': coverage_percentage,
        'coverage_ratio': coverage_ratio,
        'node_coverage_map': node_coverage_map,
        'explanation': explanation
    }
