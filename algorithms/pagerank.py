"""
Algorithm 2: PageRank Centrality for Influential User Detection.

PageRank measures the relative importance and transmission potential of nodes
based on the global link structure of the social network.
"""

import networkx as nx
import pandas as pd


def compute_pagerank(G: nx.Graph, alpha: float = 0.85, top_k: int = 5) -> dict:
    """
    Computes PageRank centrality scores for all nodes in the network.
    
    Args:
        G (nx.Graph): Social network graph.
        alpha (float): Damping factor (default: 0.85).
        top_k (int): Number of top influential users to highlight.
        
    Returns:
        dict: Full scores, ranked DataFrame, and top-k influential users.
    """
    if G.number_of_nodes() == 0:
        return {
            'scores': {},
            'top_users': [],
            'ranking_df': pd.DataFrame(columns=['Rank', 'User', 'PageRank Score', 'Connections']),
            'explanation': "Graph is empty."
        }
        
    # Compute PageRank
    scores = nx.pagerank(G, alpha=alpha, max_iter=200, tol=1e-6)
    
    # Sort nodes by score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    rows = []
    for rank, (node, score) in enumerate(sorted_items, start=1):
        rows.append({
            'Rank': rank,
            'User': node,
            'PageRank Score': round(score, 5),
            'Connections': G.degree(node)
        })
        
    ranking_df = pd.DataFrame(rows)
    top_users = [node for node, _ in sorted_items[:top_k]]
    
    explanation = (
        f"PageRank models a random information surfer transitioning across follower connections "
        f"with damping factor alpha={alpha}. Nodes with high PageRank receive links from other well-connected "
        f"influential nodes, making them super-spreaders capable of amplifying fake news rapidly."
    )
    
    return {
        'scores': scores,
        'top_users': top_users,
        'ranking_df': ranking_df,
        'explanation': explanation
    }
