"""
Graph and statistical visualization utilities using Matplotlib.
Provides clean, presentation-ready figures for Streamlit display.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def get_fixed_layout(G: nx.Graph, seed: int = 42):
    """Generates a stable spring layout for consistent visual alignment."""
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, seed=seed, k=0.45, iterations=60)


def plot_social_network(G: nx.Graph, pos=None, title="Social Network Architecture"):
    """Visualizes the initial social network topology."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    
    if pos is None:
        pos = get_fixed_layout(G)
        
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Graph is Empty", color="white", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, pos

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#64748b",
        width=1.5,
        alpha=0.6
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="#38bdf8",
        node_size=600,
        edgecolors="#ffffff",
        linewidths=1.5,
        alpha=0.95
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#0f172a"
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig, pos


def plot_propagation_graph(G: nx.Graph, pos: dict, source: str, levels: dict, title="Fake News Propagation Levels", propagation_edges=None):
    """
    Visualizes the BFS or DFS information spread.
    - Source: Bright Crimson / Red
    - Level 1: Dark Orange
    - Level 2: Amber / Yellow
    - Level 3+: Magenta / Violet
    - Unaffected: Muted Gray
    """
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=120)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    
    if pos is None:
        pos = get_fixed_layout(G)
        
    # Draw base edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#475569",
        width=1.2,
        alpha=0.4
    )
    
    # Group nodes by level
    level_colors = {
        0: "#ef4444",  # Source: Red
        1: "#f97316",  # Level 1: Orange
        2: "#eab308",  # Level 2: Yellow
        3: "#a855f7",  # Level 3: Purple
        4: "#ec4899",  # Level 4: Pink
    }
    
    unaffected = [n for n in G.nodes() if n not in levels]
    
    # Draw unaffected nodes
    if unaffected:
        nx.draw_networkx_nodes(
            G, pos, nodelist=unaffected, ax=ax,
            node_color="#64748b",
            node_size=450,
            edgecolors="#94a3b8",
            linewidths=1.0,
            alpha=0.5,
            label="Unaffected User"
        )
        
    # Draw infected levels
    max_level = max(levels.values()) if levels else 0
    for lvl in range(max_level + 1):
        lvl_nodes = [n for n, l in levels.items() if l == lvl and n in G.nodes()]
        if not lvl_nodes:
            continue
        
        color = level_colors.get(lvl, "#06b6d4")
        lbl = f"Source ({source})" if lvl == 0 else f"Propagation Step {lvl}"
        size = 800 if lvl == 0 else 550
        edge_col = "#ffffff" if lvl == 0 else "#ffffff"
        lw = 2.5 if lvl == 0 else 1.5
        
        nx.draw_networkx_nodes(
            G, pos, nodelist=lvl_nodes, ax=ax,
            node_color=color,
            node_size=size,
            edgecolors=edge_col,
            linewidths=lw,
            alpha=0.95,
            label=lbl
        )
        
    # Highlight active propagation tree edges
    if propagation_edges:
        tree_edges = [
            (u, v) for u, v in propagation_edges
            if u in G.nodes and v in G.nodes and G.has_edge(u, v)
        ]
    else:
        tree_edges = []
        for u, v in G.edges():
            if u in levels and v in levels:
                if abs(levels[u] - levels[v]) == 1:
                    tree_edges.append((u, v))
                
    if tree_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=tree_edges, ax=ax,
            edge_color="#f87171",
            width=2.4,
            alpha=0.85
        )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#ffffff"
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    legend = ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_highlighted_nodes(G: nx.Graph, pos: dict, highlighted_nodes: list, metric_name: str, highlight_color: str = "#10b981", scores: dict = None):
    """Highlights top critical nodes (PageRank influencers, Betweenness bridges, or Dominating set)."""
    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=120)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    
    if pos is None:
        pos = get_fixed_layout(G)
        
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#475569",
        width=1.3,
        alpha=0.5
    )
    
    normal_nodes = [n for n in G.nodes() if n not in highlighted_nodes]
    
    # Draw normal nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=normal_nodes, ax=ax,
        node_color="#38bdf8",
        node_size=500,
        edgecolors="#ffffff",
        linewidths=1.0,
        alpha=0.7,
        label="Regular User"
    )
    
    # Draw highlighted nodes
    if highlighted_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=highlighted_nodes, ax=ax,
            node_color=highlight_color,
            node_size=850,
            edgecolors="#ffffff",
            linewidths=2.2,
            alpha=0.98,
            label=f"Selected: {metric_name}"
        )
        
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#ffffff"
    )
    
    ax.set_title(f"Network Analysis: {metric_name}", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_containment_network(
    G_original: nx.Graph,
    pos: dict,
    source: str,
    blocked_nodes: list = None,
    cut_edges: list = None,
    affected_after: list = None,
    title="Post-Containment Social Network"
):
    """
    Visualizes network state after containment intervention:
    - Blocked nodes shown in Red Cross / dark badge
    - Cut edges shown as dashed red or omitted
    - Still infected nodes shown in Orange
    - Protected unaffected nodes shown in Emerald Green
    """
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=120)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    
    if pos is None:
        pos = get_fixed_layout(G_original)
        
    blocked_nodes = blocked_nodes or []
    cut_edges = cut_edges or []
    affected_after = affected_after or []
    
    # Remaining edges
    remaining_edges = [
        (u, v) for u, v in G_original.edges()
        if (u, v) not in cut_edges and (v, u) not in cut_edges
        and u not in blocked_nodes and v not in blocked_nodes
    ]
    
    # Draw active edges
    nx.draw_networkx_edges(
        G_original, pos, edgelist=remaining_edges, ax=ax,
        edge_color="#64748b",
        width=1.5,
        alpha=0.7
    )
    
    # Draw severed cut edges
    severed_edges = [
        (u, v) for u, v in G_original.edges()
        if (u, v) in cut_edges or (v, u) in cut_edges
    ]
    if severed_edges:
        nx.draw_networkx_edges(
            G_original, pos, edgelist=severed_edges, ax=ax,
            edge_color="#ef4444",
            width=2.5,
            style="dashed",
            alpha=0.9,
            label="Severed Cut Edge"
        )
        
    # Classify nodes
    protected_nodes = [
        n for n in G_original.nodes()
        if n not in blocked_nodes and n not in affected_after
    ]
    infected_remaining = [
        n for n in affected_after
        if n not in blocked_nodes and n != source
    ]
    
    # 1. Protected nodes (Green)
    if protected_nodes:
        nx.draw_networkx_nodes(
            G_original, pos, nodelist=protected_nodes, ax=ax,
            node_color="#10b981",
            node_size=550,
            edgecolors="#ffffff",
            linewidths=1.2,
            alpha=0.9,
            label="Protected / Contained"
        )
        
    # 2. Remaining infected nodes (Orange)
    if infected_remaining:
        nx.draw_networkx_nodes(
            G_original, pos, nodelist=infected_remaining, ax=ax,
            node_color="#f97316",
            node_size=550,
            edgecolors="#ffffff",
            linewidths=1.5,
            alpha=0.9,
            label="Affected Post-Containment"
        )
        
    # 3. Source node (Red)
    if source in G_original.nodes():
        nx.draw_networkx_nodes(
            G_original, pos, nodelist=[source], ax=ax,
            node_color="#ef4444",
            node_size=800,
            edgecolors="#ffffff",
            linewidths=2.5,
            alpha=0.98,
            label="Fake News Source"
        )
        
    # 4. Blocked nodes (Dark Crimson with 'X' marker)
    if blocked_nodes:
        nx.draw_networkx_nodes(
            G_original, pos, nodelist=blocked_nodes, ax=ax,
            node_color="#881337",
            node_shape="s",
            node_size=650,
            edgecolors="#fb7185",
            linewidths=2.0,
            alpha=0.95,
            label="Blocked User Account"
        )

    # Labels
    nx.draw_networkx_labels(
        G_original, pos, ax=ax,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#ffffff"
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_comparison_bar_chart(before_stats: dict, after_stats: dict):
    """Generates a side-by-side comparative bar chart between Before and After containment."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    
    metrics = ["Affected Users", "Unaffected Users", "Max Spread Depth"]
    before_vals = [
        before_stats.get("affected_count", 0),
        before_stats.get("unaffected_count", 0),
        before_stats.get("max_level", 0)
    ]
    after_vals = [
        after_stats.get("affected_count", 0),
        after_stats.get("unaffected_count", 0),
        after_stats.get("max_level", 0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, before_vals, width, label='Before Containment', color='#ef4444', alpha=0.85)
    rects2 = ax.bar(x + width/2, after_vals, width, label='After Containment', color='#10b981', alpha=0.85)
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='#fca5a5', fontweight='bold', fontsize=9)
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', color='#6ee7b7', fontweight='bold', fontsize=9)

    ax.set_ylabel('User Count / Depth Level', color='#f8fafc', fontsize=11, fontweight='bold')
    ax.set_title('Propagation Metrics: Before vs After Containment', color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#f8fafc', fontweight='bold', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')
    ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)
    
    for spine in ax.spines.values():
        spine.set_color('#475569')
        
    plt.tight_layout()
    return fig
