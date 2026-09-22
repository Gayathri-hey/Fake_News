"""
Graph construction, validation, and topological metric utilities using NetworkX.
"""

import os
import pandas as pd
import networkx as nx


def load_graph_from_csv(file_or_path) -> nx.Graph:
    """
    Loads a social network graph from a CSV file or file-like buffer.
    Expects columns: 'source' and 'target' (or similar column names).
    """
    try:
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                raise FileNotFoundError(f"File not found: {file_or_path}")
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Normalize column names
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)
    
    # Identify source and target columns
    src_col = None
    tgt_col = None
    for c in df.columns:
        if c in ['source', 'from', 'src', 'user1', 'node1']:
            src_col = c
        elif c in ['target', 'to', 'dst', 'user2', 'node2']:
            tgt_col = c
            
    if not src_col or not tgt_col:
        # Fallback to first two columns if present
        if len(df.columns) >= 2:
            src_col = df.columns[0]
            tgt_col = df.columns[1]
        else:
            raise ValueError("CSV must contain at least two columns representing source and target user connections.")
            
    # Clean data
    df = df.dropna(subset=[src_col, tgt_col])
    df[src_col] = df[src_col].astype(str).str.strip()
    df[tgt_col] = df[tgt_col].astype(str).str.strip()
    
    # Remove self-loops and empty strings
    df = df[(df[src_col] != '') & (df[tgt_col] != '') & (df[src_col] != df[tgt_col])]
    
    if len(df) == 0:
        raise ValueError("The provided CSV has no valid edges.")
        
    G = nx.Graph()
    for _, row in df.iterrows():
        u, v = row[src_col], row[tgt_col]
        G.add_edge(u, v, capacity=1.0)  # Default unit capacity for flow networks
        
    return G


def generate_default_social_network() -> nx.Graph:
    """
    Generates a 30-node modular social network representing 3 distinct communities
    with hubs, bridge nodes, and bottlenecks.
    """
    edges = [
        ("User1", "User2"), ("User1", "User3"), ("User1", "User4"), ("User1", "User5"),
        ("User2", "User3"), ("User2", "User6"), ("User3", "User7"), ("User4", "User5"),
        ("User4", "User8"), ("User5", "User6"), ("User6", "User7"), ("User6", "User9"),
        ("User7", "User8"), ("User7", "User10"), ("User8", "User9"), ("User9", "User10"),
        ("User7", "User11"), ("User8", "User12"), ("User10", "User13"),
        ("User11", "User12"), ("User11", "User14"), ("User11", "User15"),
        ("User12", "User14"), ("User12", "User16"), ("User13", "User14"), ("User13", "User17"),
        ("User14", "User15"), ("User14", "User18"), ("User15", "User16"), ("User15", "User19"),
        ("User16", "User17"), ("User16", "User20"), ("User17", "User18"), ("User18", "User19"),
        ("User19", "User20"),
        ("User15", "User21"), ("User18", "User22"), ("User20", "User23"),
        ("User21", "User22"), ("User21", "User24"), ("User21", "User25"),
        ("User22", "User24"), ("User22", "User26"), ("User23", "User25"), ("User23", "User27"),
        ("User24", "User26"), ("User24", "User28"), ("User25", "User27"), ("User25", "User29"),
        ("User26", "User28"), ("User26", "User30"), ("User27", "User29"), ("User27", "User30"),
        ("User28", "User29"), ("User28", "User30"), ("User29", "User30"),
        ("User3", "User12"), ("User9", "User18"), ("User14", "User26"), ("User2", "User11"),
        ("User17", "User25")
    ]
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v, capacity=1.0)
    return G


def get_graph_metrics(G: nx.Graph) -> dict:
    """Computes fundamental graph metrics for summary and reporting."""
    if G.number_of_nodes() == 0:
        return {
            'num_nodes': 0, 'num_edges': 0, 'density': 0.0,
            'avg_degree': 0.0, 'num_components': 0, 'is_connected': False,
            'diameter': 'N/A'
        }
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    num_components = nx.number_connected_components(G)
    is_connected = nx.is_connected(G)
    
    if is_connected:
        diameter = nx.diameter(G)
    else:
        # Diameter of largest component
        largest_cc = max(nx.connected_components(G), key=len)
        sub_g = G.subgraph(largest_cc)
        diameter = f"{nx.diameter(sub_g)} (Largest Component)"
        
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': round(density, 4),
        'avg_degree': round(avg_degree, 2),
        'num_components': num_components,
        'is_connected': is_connected,
        'diameter': diameter
    }


def validate_source_target(G: nx.Graph, source: str, target: str = None) -> tuple[bool, str]:
    """Validates source and target user selections on graph G."""
    if G.number_of_nodes() == 0:
        return False, "Graph is empty."
    if source not in G.nodes:
        return False, f"Source user '{source}' does not exist in the network."
    if target is not None:
        if target not in G.nodes:
            return False, f"Target user '{target}' does not exist in the network."
        if source == target:
            return False, "Source user and Target user cannot be the same user."
    return True, "Valid"
