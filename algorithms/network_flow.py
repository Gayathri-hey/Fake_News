"""
Algorithm 5: Network Flow and Minimum Cut for Critical Connection Containment.

Utilizes the Max-Flow Min-Cut Theorem to find the minimum number of critical
transmission edges whose removal completely disconnects the fake news source
from reaching a sensitive target user or community.
"""

import networkx as nx


def compute_minimum_cut_containment(G: nx.Graph, source: str, target: str) -> dict:
    """
    Computes the Minimum Edge Cut separating source user from target user.
    
    Args:
        G (nx.Graph): Social network graph.
        source (str): Fake news source user node.
        target (str): Sensitive target user node to protect.
        
    Returns:
        dict: Cut value, critical edges, partition sets, and academic explanation.
    """
    if source not in G.nodes:
        return {'success': False, 'error': f"Source '{source}' is not present in graph."}
    if target not in G.nodes:
        return {'success': False, 'error': f"Target '{target}' is not present in graph."}
    if source == target:
        return {'success': False, 'error': "Source and Target cannot be the identical user."}
        
    # Check if a path exists
    if not nx.has_path(G, source, target):
        return {
            'success': True,
            'source': source,
            'target': target,
            'cut_value': 0.0,
            'critical_edges': [],
            'partition_source_size': len(nx.node_connected_component(G, source)),
            'partition_target_size': len(nx.node_connected_component(G, target)),
            'already_disconnected': True,
            'explanation': f"Source '{source}' and Target '{target}' are already in disconnected network components. No path exists."
        }
        
    # Ensure capacity attribute on edges
    for u, v in G.edges():
        if 'capacity' not in G[u][v]:
            G[u][v]['capacity'] = 1.0
            
    # Compute Minimum Cut using NetworkX (Edmonds-Karp / Boykov-Kolmogorov algorithm)
    cut_value, partition = nx.minimum_cut(G, source, target, capacity='capacity')
    reachable, non_reachable = partition
    
    # Identify critical cut edges bridging reachable and non-reachable partitions
    critical_edges = []
    for u in reachable:
        for v in G.neighbors(u):
            if v in non_reachable:
                critical_edges.append((u, v))
                
    explanation = (
        f"By the Max-Flow Min-Cut Theorem, the minimum transmission capacity required to isolate "
        f"Target '{target}' from Source '{source}' is {cut_value}. Severing the {len(critical_edges)} critical "
        f"edge(s) {critical_edges} completely eliminates all direct and indirect propagation corridors "
        f"between the fake news source and the protected target."
    )
    
    return {
        'success': True,
        'source': source,
        'target': target,
        'cut_value': float(cut_value),
        'critical_edges': critical_edges,
        'partition_source_size': len(reachable),
        'partition_target_size': len(non_reachable),
        'already_disconnected': False,
        'explanation': explanation
    }
