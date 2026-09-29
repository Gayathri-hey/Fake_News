"""
Algorithm 1: Breadth-First Search (BFS) and Depth-First Search (DFS)
Fake News Propagation Simulation.

Simulates how misinformation cascades through a social network:
- BFS: Level-by-level (wavefront/epidemic) broadcast cascade.
- DFS: Sequential (word-of-mouth/chain) deep rumor cascade.
"""

from collections import deque
import networkx as nx


def simulate_bfs_propagation(
    G: nx.Graph,
    source: str,
    max_steps: int = None,
    blocked_nodes: list = None,
    blocked_edges: list = None
) -> dict:
    """
    Simulates information propagation using Breadth-First Search (BFS).
    Models simultaneous level-by-level broadcast cascades.
    
    Args:
        G (nx.Graph): Social network graph.
        source (str): Initial node where fake news originates.
        max_steps (int, optional): Maximum propagation depth/levels to simulate.
        blocked_nodes (list, optional): Nodes to remove/block from propagation.
        blocked_edges (list, optional): Edges to remove/block from propagation.
        
    Returns:
        dict: Complete propagation metrics and step-by-step cascades.
    """
    blocked_nodes = set(blocked_nodes or [])
    blocked_edges = set(tuple(sorted(e)) for e in (blocked_edges or []))
    
    total_nodes = G.number_of_nodes()
    
    # Handle edge case where source is invalid or blocked
    if source not in G.nodes:
        return {
            'success': False,
            'algorithm': 'BFS',
            'error': f"Source '{source}' is not in the network.",
            'total_users': total_nodes,
            'affected_count': 0,
            'unaffected_count': total_nodes,
            'propagation_percentage': 0.0,
            'levels': {},
            'level_groups': {},
            'propagation_order': [],
            'propagation_edges': [],
            'affected_users': [],
            'unaffected_users': list(G.nodes()),
            'max_level': 0
        }
        
    if source in blocked_nodes:
        return {
            'success': True,
            'algorithm': 'BFS',
            'message': f"Source user '{source}' is blocked. Zero propagation occurred.",
            'total_users': total_nodes,
            'affected_count': 0,
            'unaffected_count': total_nodes,
            'propagation_percentage': 0.0,
            'levels': {},
            'level_groups': {},
            'propagation_order': [],
            'propagation_edges': [],
            'affected_users': [],
            'unaffected_users': list(G.nodes()),
            'max_level': 0
        }
    
    # Standard BFS Queue: stores (node, current_level)
    queue = deque([(source, 0)])
    visited = {source}
    levels = {source: 0}
    level_groups = {0: [source]}
    propagation_order = [source]
    propagation_edges = []
    
    while queue:
        current_node, current_lvl = queue.popleft()
        
        # Check if max step depth is reached
        if max_steps is not None and current_lvl >= max_steps:
            continue
            
        next_lvl = current_lvl + 1
        
        for neighbor in sorted(G.neighbors(current_node)):
            # Check if neighbor is blocked
            if neighbor in blocked_nodes:
                continue
                
            # Check if connecting edge is blocked / severed
            edge = tuple(sorted((current_node, neighbor)))
            if edge in blocked_edges:
                continue
                
            if neighbor not in visited:
                visited.add(neighbor)
                levels[neighbor] = next_lvl
                propagation_order.append(neighbor)
                propagation_edges.append((current_node, neighbor))
                
                if next_lvl not in level_groups:
                    level_groups[next_lvl] = []
                level_groups[next_lvl].append(neighbor)
                
                queue.append((neighbor, next_lvl))
                
    affected_count = len(visited)
    unaffected_users = [n for n in G.nodes() if n not in visited and n not in blocked_nodes]
    unaffected_count = len(unaffected_users)
    propagation_percentage = round((affected_count / total_nodes) * 100, 2) if total_nodes > 0 else 0.0
    max_level = max(levels.values()) if levels else 0
    
    return {
        'success': True,
        'algorithm': 'BFS',
        'source': source,
        'total_users': total_nodes,
        'affected_count': affected_count,
        'unaffected_count': unaffected_count,
        'propagation_percentage': propagation_percentage,
        'levels': levels,
        'level_groups': level_groups,
        'propagation_order': propagation_order,
        'propagation_edges': propagation_edges,
        'affected_users': list(visited),
        'unaffected_users': unaffected_users,
        'max_level': max_level
    }


def simulate_dfs_propagation(
    G: nx.Graph,
    source: str,
    max_steps: int = None,
    blocked_nodes: list = None,
    blocked_edges: list = None
) -> dict:
    """
    Simulates information propagation using Depth-First Search (DFS).
    Models deep sequential chain cascades (word-of-mouth forwarding).
    
    Args:
        G (nx.Graph): Social network graph.
        source (str): Initial node where fake news originates.
        max_steps (int, optional): Maximum propagation depth/steps to explore along any path.
        blocked_nodes (list, optional): Nodes to remove/block from propagation.
        blocked_edges (list, optional): Edges to remove/block from propagation.
        
    Returns:
        dict: Complete propagation metrics and step-by-step cascades.
    """
    blocked_nodes = set(blocked_nodes or [])
    blocked_edges = set(tuple(sorted(e)) for e in (blocked_edges or []))
    
    total_nodes = G.number_of_nodes()
    
    # Handle edge case where source is invalid or blocked
    if source not in G.nodes:
        return {
            'success': False,
            'algorithm': 'DFS',
            'error': f"Source '{source}' is not in the network.",
            'total_users': total_nodes,
            'affected_count': 0,
            'unaffected_count': total_nodes,
            'propagation_percentage': 0.0,
            'levels': {},
            'level_groups': {},
            'propagation_order': [],
            'propagation_edges': [],
            'affected_users': [],
            'unaffected_users': list(G.nodes()),
            'max_level': 0
        }
        
    if source in blocked_nodes:
        return {
            'success': True,
            'algorithm': 'DFS',
            'message': f"Source user '{source}' is blocked. Zero propagation occurred.",
            'total_users': total_nodes,
            'affected_count': 0,
            'unaffected_count': total_nodes,
            'propagation_percentage': 0.0,
            'levels': {},
            'level_groups': {},
            'propagation_order': [],
            'propagation_edges': [],
            'affected_users': [],
            'unaffected_users': list(G.nodes()),
            'max_level': 0
        }
    
    # Iterative DFS Stack: stores (current_node, current_depth, neighbors_iterator)
    stack = [(source, 0, iter(sorted(G.neighbors(source))))]
    visited = {source}
    levels = {source: 0}
    level_groups = {0: [source]}
    propagation_order = [source]
    propagation_edges = []
    
    while stack:
        current_node, current_lvl, neighbors_iter = stack[-1]
        
        # Check if max step depth is reached on this branch
        if max_steps is not None and current_lvl >= max_steps:
            stack.pop()
            continue
            
        found_unvisited = False
        for neighbor in neighbors_iter:
            if neighbor in blocked_nodes:
                continue
                
            edge = tuple(sorted((current_node, neighbor)))
            if edge in blocked_edges:
                continue
                
            if neighbor not in visited:
                visited.add(neighbor)
                next_lvl = current_lvl + 1
                levels[neighbor] = next_lvl
                propagation_order.append(neighbor)
                propagation_edges.append((current_node, neighbor))
                
                if next_lvl not in level_groups:
                    level_groups[next_lvl] = []
                level_groups[next_lvl].append(neighbor)
                
                stack.append((neighbor, next_lvl, iter(sorted(G.neighbors(neighbor)))))
                found_unvisited = True
                break
                
        if not found_unvisited:
            stack.pop()
            
    affected_count = len(visited)
    unaffected_users = [n for n in G.nodes() if n not in visited and n not in blocked_nodes]
    unaffected_count = len(unaffected_users)
    propagation_percentage = round((affected_count / total_nodes) * 100, 2) if total_nodes > 0 else 0.0
    max_level = max(levels.values()) if levels else 0
    
    return {
        'success': True,
        'algorithm': 'DFS',
        'source': source,
        'total_users': total_nodes,
        'affected_count': affected_count,
        'unaffected_count': unaffected_count,
        'propagation_percentage': propagation_percentage,
        'levels': levels,
        'level_groups': level_groups,
        'propagation_order': propagation_order,
        'propagation_edges': propagation_edges,
        'affected_users': list(visited),
        'unaffected_users': unaffected_users,
        'max_level': max_level
    }


def simulate_propagation(
    G: nx.Graph,
    source: str,
    algorithm: str = "BFS",
    max_steps: int = None,
    blocked_nodes: list = None,
    blocked_edges: list = None
) -> dict:
    """
    Unified propagation simulator allowing dynamic choice between BFS and DFS.
    
    Args:
        G (nx.Graph): Social network graph.
        source (str): Origin node of misinformation.
        algorithm (str): 'BFS' or 'DFS' (case-insensitive).
        max_steps (int, optional): Depth/hop limit.
        blocked_nodes (list, optional): Quarantined nodes.
        blocked_edges (list, optional): Severed edges.
        
    Returns:
        dict: Complete propagation results.
    """
    algo_clean = (algorithm or "BFS").upper()
    if "DFS" in algo_clean:
        return simulate_dfs_propagation(
            G, source, max_steps=max_steps, blocked_nodes=blocked_nodes, blocked_edges=blocked_edges
        )
    return simulate_bfs_propagation(
        G, source, max_steps=max_steps, blocked_nodes=blocked_nodes, blocked_edges=blocked_edges
    )
