"""
Algorithm 3: Betweenness Centrality for Bridge and Gatekeeper User Detection.

Betweenness centrality identifies nodes that act as structural bridges between
different communities and sub-clusters in the social network.
"""

import networkx as nx
import pandas as pd


def compute_betweenness_centrality(G: nx.Graph, top_k: int = 5) -> dict:
    """
    Computes Betweenness Centrality scores for all nodes in the network.
    
    Args:
        G (nx.Graph): Social network graph.
        top_k (int): Number of top bridge users to highlight.
        
    Returns:
        dict: Full scores, ranked DataFrame, and top-k bridge users.
    """
    if G.number_of_nodes() == 0:
        return {
            'scores': {},
            'top_users': [],
            'ranking_df': pd.DataFrame(columns=['Rank', 'User', 'Betweenness Score', 'Connections']),
            'explanation': "Graph is empty."
        }
        
    # Compute Betweenness Centrality
    scores = nx.betweenness_centrality(G, normalized=True)
    
    # Sort descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    rows = []
    for rank, (node, score) in enumerate(sorted_items, start=1):
        rows.append({
            'Rank': rank,
            'User': node,
            'Betweenness Score': round(score, 5),
            'Connections': G.degree(node)
        })
        
    ranking_df = pd.DataFrame(rows)
    top_users = [node for node, _ in sorted_items[:top_k]]
    
    explanation = (
        "Betweenness Centrality measures the proportion of all-pairs shortest paths in the network "
        "that pass through a specific user node. High-betweenness users function as informational "
        "'bridges' and bottlenecks connecting disparate network clusters. Monitoring or blocking these "
        "gatekeepers effectively halts cross-community misinformation leakage."
    )
    
    return {
        'scores': scores,
        'top_users': top_users,
        'ranking_df': ranking_df,
        'explanation': explanation
    }
