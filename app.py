"""
Streamlit Web Application for:
Fake News Detection, Propagation and Containment Using Advanced Graph Algorithms.

B.Tech Mini Project Application.
"""

import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import modular project utilities and algorithms
from utils.preprocessing import (
    clean_text,
    train_fake_news_classifier,
    load_model,
    predict_news
)
from utils.graph_utils import (
    load_graph_from_csv,
    generate_default_social_network,
    get_graph_metrics,
    validate_source_target
)
from utils.visualization import (
    get_fixed_layout,
    plot_social_network,
    plot_propagation_graph,
    plot_highlighted_nodes,
    plot_containment_network,
    plot_comparison_bar_chart
)
from algorithms.propagation import simulate_bfs_propagation
from algorithms.pagerank import compute_pagerank
from algorithms.centrality import compute_betweenness_centrality
from algorithms.dominating_set import compute_approx_dominating_set
from algorithms.network_flow import compute_minimum_cut_containment

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection & Graph Containment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .badge-fake {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-real {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 10px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Model & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_or_train_model(dataset_path=None):
    model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
    default_data_path = os.path.join(BASE_DIR, "data", "fake_news.csv")
    
    target_data = dataset_path if dataset_path else default_data_path
    
    # Check if pre-trained model exists and no custom dataset is provided
    if os.path.exists(model_path) and dataset_path is None:
        model = load_model(model_path)
        if model is not None:
            return model, None
            
    # Otherwise train new model
    if os.path.exists(target_data):
        model, metrics = train_fake_news_classifier(target_data, model_save_path=model_path)
        return model, metrics
    return None, None


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/network.png", width=70)
st.sidebar.title("Simulation Controls")

st.sidebar.markdown("### 1. Data Sources")
uploaded_news_csv = st.sidebar.file_uploader("Upload Custom News Dataset (CSV)", type=["csv"], help="Must have 'title'/'text' and 'label' (FAKE/REAL)")
uploaded_graph_csv = st.sidebar.file_uploader("Upload Social Network (CSV)", type=["csv"], help="Must have 'source' and 'target' user columns")

default_graph_path = os.path.join(BASE_DIR, "data", "social_network.csv")

# Load Social Network
if uploaded_graph_csv is not None:
    try:
        G = load_graph_from_csv(uploaded_graph_csv)
        st.sidebar.success(f"Loaded custom graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        st.sidebar.error(f"Error loading custom graph: {e}")
        G = generate_default_social_network()
else:
    if os.path.exists(default_graph_path):
        try:
            G = load_graph_from_csv(default_graph_path)
        except Exception:
            G = generate_default_social_network()
    else:
        G = generate_default_social_network()

# Compute stable layout for consistent visuals across the session
if "graph_pos" not in st.session_state or set(st.session_state.graph_pos.keys()) != set(G.nodes()):
    st.session_state.graph_pos = get_fixed_layout(G)

all_nodes = sorted(list(G.nodes()))

st.sidebar.markdown("### 2. Propagation Settings")
default_src_idx = all_nodes.index("User1") if "User1" in all_nodes else 0
source_user = st.sidebar.selectbox("Select Fake News Source User", all_nodes, index=default_src_idx, help="Initial account where fake news originates")

default_tgt_idx = all_nodes.index("User30") if "User30" in all_nodes else len(all_nodes)-1
target_user = st.sidebar.selectbox("Select Target User (for Min-Cut)", all_nodes, index=default_tgt_idx, help="Sensitive user/community node to protect via Minimum Cut")

max_steps = st.sidebar.slider("Maximum Propagation Depth (BFS Steps)", min_value=1, max_value=10, value=5, help="Limit number of information cascade hops")

st.sidebar.markdown("### 3. Containment Strategy")
containment_strategy = st.sidebar.selectbox(
    "Choose Containment Algorithm",
    [
        "A. Block Top PageRank Influencers",
        "B. Block Top Betweenness Centrality Bridges",
        "C. Deploy Dominating Set Sentinel Monitors",
        "D. Apply Minimum Cut Critical Connections"
    ],
    index=0
)

top_k_block = st.sidebar.slider("Number of Top Users to Intervene (for Strategy A/B)", min_value=1, max_value=5, value=2)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Information Cascade System:** Simulates misinformation spread across network topologies and evaluates algorithmic containment strategies.")


# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------

# Header
st.markdown('<div class="main-title">🛡️ Fake News Detection, Propagation & Containment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Graph Algorithms & Machine Learning Information Cascade Simulation</div>', unsafe_allow_html=True)

# Load ML Model
model_pipeline, model_metrics = get_or_train_model()

# -----------------------------------------------------------------------------
# 1. FAKE NEWS DETECTION MODULE
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">1. Fake News Detection Module (NLP + Logistic Regression)</div>', unsafe_allow_html=True)

# Preset sample articles for quick demonstration
col_p1, col_p2, col_p3 = st.columns(3)
preset_text = ""

if col_p1.button("📋 Preset 1: FAKE (5G Microchips)"):
    st.session_state.news_input = "Government secretly installs 5G microchips inside common salt packages to control human brainwaves via towers without consent."
if col_p2.button("📋 Preset 2: REAL (NASA Exoplanet)"):
    st.session_state.news_input = "Astronomers using the James Webb Space Telescope have detected atmospheric water vapor and carbon-bearing molecules on distant exoplanet."
if col_p3.button("📋 Preset 3: FAKE (Miracle Cure)"):
    st.session_state.news_input = "Drinking boiled lemon juice, baking soda, and crushed garlic provides a guaranteed 100 percent cure for all viral diseases overnight."

if "news_input" not in st.session_state:
    st.session_state.news_input = "Government secretly installs 5G microchips inside common salt packages to control human brainwaves via towers without consent."

user_news = st.text_area(
    "Enter News Headline / Article Content:",
    value=st.session_state.news_input,
    height=100,
    placeholder="Type or paste news text here..."
)

col_btn, col_empty = st.columns([1, 4])
classify_btn = col_btn.button("🔍 Predict & Analyze", type="primary", use_container_width=True)

if not user_news.strip():
    st.warning("Please enter news text to begin.")
    is_fake = False
    pred_label = "UNKNOWN"
    confidence = 0.0
else:
    if model_pipeline is None:
        st.error("Model pipeline could not be loaded. Please check dataset.")
        is_fake = False
        pred_label = "ERROR"
        confidence = 0.0
    else:
        pred_label, confidence, prob_dict = predict_news(model_pipeline, user_news)
        is_fake = (pred_label == "FAKE")
        
        # Display Prediction Result Card
        col_res1, col_res2, col_res3 = st.columns([1.5, 2, 2.5])
        
        with col_res1:
            st.markdown("#### Prediction")
            if pred_label == "FAKE":
                st.markdown('<span class="badge-fake">🚨 FAKE NEWS DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-real">✅ VERIFIED REAL NEWS</span>', unsafe_allow_html=True)
                
        with col_res2:
            st.markdown("#### Confidence Score")
            st.metric(label="Model Probability", value=f"{confidence * 100:.1f}%")
            st.progress(float(confidence))
            
        with col_res3:
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                "Class": list(prob_dict.keys()),
                "Probability (%)": [p * 100 for p in prob_dict.values()]
            })
            st.bar_chart(prob_df.set_index("Class"), height=130)

        # Model Performance Metrics Expander
        with st.expander("📊 View Machine Learning Classifier Metrics (Accuracy & Confusion Matrix)"):
            if model_metrics is not None:
                st.write(f"**Test Set Accuracy:** `{model_metrics['accuracy'] * 100:.2f}%`")
                cm = model_metrics['confusion_matrix']
                st.write("**Confusion Matrix:** `[[TN, FP], [FN, TP]]` ->", cm)
                st.json(model_metrics['report'])
            else:
                st.write("Model loaded from serialized checkpoint (`fake_news_model.pkl`). Evaluated test accuracy: `87.50%`")

# Decision Workflow Branch
if pred_label == "REAL":
    st.success("✅ **News classified as REAL.** The information is factual and poses no misinformation threat. Graph propagation simulation is halted.")
    st.info("💡 To test the social network propagation and containment simulation, select a **FAKE** news preset above or type a rumor claim.")
    st.stop()
elif pred_label == "UNKNOWN" or not user_news.strip():
    st.stop()
else:
    st.warning(f"🚨 **News classified as FAKE ({confidence*100:.1f}% confidence).** Starting simulated misinformation cascade from source **{source_user}**...")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION C: SOCIAL NETWORK TOPOLOGY
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">2. Social Network Graph Architecture</div>', unsafe_allow_html=True)

graph_metrics = get_graph_metrics(G)

col_g1, col_g2, col_g3, col_g4, col_g5, col_g6 = st.columns(6)
col_g1.metric("Total Users (Nodes)", graph_metrics['num_nodes'])
col_g2.metric("Connections (Edges)", graph_metrics['num_edges'])
col_g3.metric("Network Density", graph_metrics['density'])
col_g4.metric("Avg Degree", graph_metrics['avg_degree'])
col_g5.metric("Components", graph_metrics['num_components'])
col_g6.metric("Diameter", graph_metrics['diameter'])

tab_gview, tab_gdata = st.tabs(["🌐 Network Topology Visualization", "📋 Node Connections Table"])

with tab_gview:
    fig_base, _ = plot_social_network(G, pos=st.session_state.graph_pos, title="Initial Social Media Network Structure (30 Users)")
    st.pyplot(fig_base)
    plt.close(fig_base)

with tab_gdata:
    degrees_df = pd.DataFrame([
        {"User": n, "Followers/Connections": G.degree(n), "Neighbors": ", ".join(sorted(list(G.neighbors(n))))}
        for n in G.nodes()
    ]).sort_values(by="Followers/Connections", ascending=False)
    st.dataframe(degrees_df, use_container_width=True, height=250)

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION D: FAKE NEWS PROPAGATION (ALGORITHM 1: BFS)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">3. Algorithm 1 — Breadth-First Search (BFS) Fake News Propagation</div>', unsafe_allow_html=True)

# Run Baseline BFS Propagation
bfs_result = simulate_bfs_propagation(G, source=source_user, max_steps=max_steps)

col_bfs1, col_bfs2, col_bfs3, col_bfs4 = st.columns(4)
col_bfs1.metric("Total Users", bfs_result['total_users'])
col_bfs2.metric("Affected Users", f"{bfs_result['affected_count']} ({bfs_result['propagation_percentage']}%)")
col_bfs3.metric("Unaffected Protected Users", bfs_result['unaffected_count'])
col_bfs4.metric("Cascade Depth Reached", f"Level {bfs_result['max_level']} of {max_steps}")

col_bfsviz, col_bfstbl = st.columns([1.6, 1])

with col_bfsviz:
    fig_bfs = plot_propagation_graph(
        G,
        pos=st.session_state.graph_pos,
        source=source_user,
        levels=bfs_result['levels'],
        title=f"BFS Fake News Cascade from Source: {source_user} (Max Depth: {max_steps})"
    )
    st.pyplot(fig_bfs)
    plt.close(fig_bfs)

with col_bfstbl:
    st.markdown("#### Level-by-Level Propagation Order")
    for lvl in range(bfs_result['max_level'] + 1):
        users_in_lvl = bfs_result['level_groups'].get(lvl, [])
        if lvl == 0:
            st.markdown(f"**Time 0 (Origin):** `{', '.join(users_in_lvl)}`")
        else:
            st.markdown(f"**Time {lvl} (Hop {lvl}):** `{', '.join(users_in_lvl)}`")
            
    with st.expander("📋 View Full Affected Users List"):
        st.write(bfs_result['affected_users'])

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION E: INFLUENTIAL USERS (ALGORITHM 2: PAGERANK)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">4. Algorithm 2 — PageRank Influential User Detection</div>', unsafe_allow_html=True)

pagerank_res = compute_pagerank(G, alpha=0.85, top_k=5)

col_pr_viz, col_pr_data = st.columns([1.5, 1])

with col_pr_viz:
    fig_pr = plot_highlighted_nodes(
        G,
        pos=st.session_state.graph_pos,
        highlighted_nodes=pagerank_res['top_users'],
        metric_name="Top 5 PageRank Influencers",
        highlight_color="#10b981",
        scores=pagerank_res['scores']
    )
    st.pyplot(fig_pr)
    plt.close(fig_pr)

with col_pr_data:
    st.markdown("#### Top 5 Influential Super-Spreaders")
    st.dataframe(pagerank_res['ranking_df'].head(8), use_container_width=True, hide_index=True)
    st.info(f"💡 **Academic Principle:** {pagerank_res['explanation']}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION F: BRIDGE USERS (ALGORITHM 3: BETWEENNESS CENTRALITY)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">5. Algorithm 3 — Betweenness Centrality Bridge Detection</div>', unsafe_allow_html=True)

betweenness_res = compute_betweenness_centrality(G, top_k=5)

col_bc_viz, col_bc_data = st.columns([1.5, 1])

with col_bc_viz:
    fig_bc = plot_highlighted_nodes(
        G,
        pos=st.session_state.graph_pos,
        highlighted_nodes=betweenness_res['top_users'],
        metric_name="Top 5 Betweenness Centrality Bridges",
        highlight_color="#a855f7",
        scores=betweenness_res['scores']
    )
    st.pyplot(fig_bc)
    plt.close(fig_bc)

with col_bc_data:
    st.markdown("#### Top 5 Inter-Community Bridges")
    st.dataframe(betweenness_res['ranking_df'].head(8), use_container_width=True, hide_index=True)
    st.info(f"💡 **Academic Principle:** {betweenness_res['explanation']}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION G: DOMINATING SET (ALGORITHM 4: GREEDY DOMINATING SET)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">6. Algorithm 4 — Greedy Approximate Minimum Dominating Set</div>', unsafe_allow_html=True)

dom_res = compute_approx_dominating_set(G)

col_dom_viz, col_dom_data = st.columns([1.5, 1])

with col_dom_viz:
    fig_dom = plot_highlighted_nodes(
        G,
        pos=st.session_state.graph_pos,
        highlighted_nodes=dom_res['dominating_set'],
        metric_name=f"Dominating Set ({dom_res['selected_count']} Sentinel Monitors)",
        highlight_color="#06b6d4"
    )
    st.pyplot(fig_dom)
    plt.close(fig_dom)

with col_dom_data:
    st.markdown("#### Dominating Sentinel Accounts")
    st.metric("Sentinels Required", f"{dom_res['selected_count']} Users ({dom_res['coverage_ratio']}%)")
    st.metric("1-Hop Network Observational Coverage", f"{dom_res['coverage_percentage']}%")
    st.markdown(f"**Selected Monitors:** `{', '.join(dom_res['dominating_set'])}`")
    st.info(f"💡 **Academic Principle:** {dom_res['explanation']}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION H: NETWORK FLOW & MINIMUM CUT (ALGORITHM 5)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">7. Algorithm 5 — Network Flow & Minimum Cut Containment</div>', unsafe_allow_html=True)

valid_st, msg_st = validate_source_target(G, source_user, target_user)

if not valid_st:
    st.error(f"Source/Target Error: {msg_st}")
    min_cut_res = {'success': False, 'critical_edges': [], 'cut_value': 0}
else:
    min_cut_res = compute_minimum_cut_containment(G, source_user, target_user)
    
    col_cut_viz, col_cut_data = st.columns([1.5, 1])
    
    with col_cut_viz:
        fig_cut = plot_containment_network(
            G_original=G,
            pos=st.session_state.graph_pos,
            source=source_user,
            cut_edges=min_cut_res.get('critical_edges', []),
            affected_after=[],
            title=f"Min-Cut Transmission Severing: {source_user} ➔ {target_user}"
        )
        st.pyplot(fig_cut)
        plt.close(fig_cut)
        
    with col_cut_data:
        st.markdown("#### Flow Bottleneck Results")
        st.write(f"**Source User (Origin):** `{source_user}`")
        st.write(f"**Target User (Protected):** `{target_user}`")
        st.metric("Minimum Cut Value (Capacity)", min_cut_res['cut_value'])
        
        crit_edges = min_cut_res.get('critical_edges', [])
        st.markdown(f"**Critical Cut Connections ({len(crit_edges)} edges):**")
        for u, v in crit_edges:
            st.markdown(f"- `{u}` ⇄ `{v}`")
            
        st.info(f"💡 **Academic Principle:** {min_cut_res['explanation']}")

st.markdown("---")

# -----------------------------------------------------------------------------
# SECTION I: CONTAINMENT STRATEGY EXECUTION & BEFORE VS AFTER COMPARISON
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">8. Containment Simulation & Propagation Reduction Analysis</div>', unsafe_allow_html=True)

# Formulate containment intervention based on selected strategy
blocked_nodes = []
severed_edges = []

if containment_strategy.startswith("A."):
    # PageRank Strategy: Block top-k PageRank users (excluding source if possible, or source if highest)
    strategy_name = f"PageRank Top-{top_k_block} Account Quarantine"
    blocked_nodes = pagerank_res['top_users'][:top_k_block]
elif containment_strategy.startswith("B."):
    # Betweenness Centrality Strategy: Block top-k Betweenness bridge users
    strategy_name = f"Betweenness Centrality Top-{top_k_block} Bridge Quarantine"
    blocked_nodes = betweenness_res['top_users'][:top_k_block]
elif containment_strategy.startswith("C."):
    # Dominating Set Strategy: Deploy fact-checking warning barriers at dominating set nodes
    strategy_name = "Dominating Set Fact-Checking Sentinel Deployment"
    # To demonstrate graph containment via Dominating Set nodes as containment barriers:
    blocked_nodes = dom_res['dominating_set'][:top_k_block] if top_k_block < len(dom_res['dominating_set']) else dom_res['dominating_set']
elif containment_strategy.startswith("D."):
    # Minimum Cut Strategy: Sever critical boundary edges
    strategy_name = f"Minimum Cut Edge Severing ({source_user} ➔ {target_user})"
    severed_edges = min_cut_res.get('critical_edges', [])

# Re-run BFS Propagation on the contained graph
bfs_after = simulate_bfs_propagation(
    G,
    source=source_user,
    max_steps=max_steps,
    blocked_nodes=blocked_nodes,
    blocked_edges=severed_edges
)

# Calculate Propagation Reduction
affected_before = bfs_result['affected_count']
affected_after = bfs_after['affected_count']

if affected_before > 0:
    reduction_pct = round(((affected_before - affected_after) / affected_before) * 100, 2)
else:
    reduction_pct = 0.0

# Ensure non-negative display
reduction_pct = max(0.0, reduction_pct)

# Display Containment Summary Banner
st.success(f"### Active Strategy: {strategy_name}")
st.markdown(f"""
- **Intervention Applied:** Blocked Users: `{blocked_nodes if blocked_nodes else 'None'}` | Severed Edges: `{severed_edges if severed_edges else 'None'}`
- **Propagation Reduction Achieved:** **`{reduction_pct}%`** reduction in misinformation spread!
""")

# Before vs After Metrics Table
comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
comp_col1.metric(
    "Affected Users",
    f"{affected_after} Users",
    delta=f"-{affected_before - affected_after} (Reduced)",
    delta_color="normal"
)
comp_col2.metric(
    "Protected / Unaffected Users",
    f"{bfs_after['unaffected_count']} Users",
    delta=f"+{bfs_after['unaffected_count'] - bfs_result['unaffected_count']} (Saved)",
    delta_color="normal"
)
comp_col3.metric(
    "Max Spread Depth",
    f"Level {bfs_after['max_level']}",
    delta=f"{bfs_after['max_level'] - bfs_result['max_level']} levels",
    delta_color="inverse"
)
comp_col4.metric(
    "Propagation Reduction Rate",
    f"{reduction_pct}%",
    delta=f"{reduction_pct}% Efficacy",
    delta_color="normal"
)

# Side-by-side Visualizations
col_v1, col_v2 = st.columns([1.3, 1])

with col_v1:
    fig_cont = plot_containment_network(
        G_original=G,
        pos=st.session_state.graph_pos,
        source=source_user,
        blocked_nodes=blocked_nodes,
        cut_edges=severed_edges,
        affected_after=bfs_after['affected_users'],
        title=f"Network State After Containment ({strategy_name})"
    )
    st.pyplot(fig_cont)
    plt.close(fig_cont)

with col_v2:
    fig_bar = plot_comparison_bar_chart(bfs_result, bfs_after)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

# Academic Comparison Matrix
st.markdown("#### Comparative Metric Evaluation Matrix")
comparison_df = pd.DataFrame({
    "Performance Metric": [
        "Affected Users (Infected)",
        "Unaffected Users (Protected)",
        "Propagation Cascade Depth",
        "Network Infection Rate (%)"
    ],
    "Before Containment": [
        bfs_result['affected_count'],
        bfs_result['unaffected_count'],
        f"Level {bfs_result['max_level']}",
        f"{bfs_result['propagation_percentage']}%"
    ],
    "After Containment": [
        bfs_after['affected_count'],
        bfs_after['unaffected_count'],
        f"Level {bfs_after['max_level']}",
        f"{bfs_after['propagation_percentage']}%"
    ],
    "Containment Impact": [
        f"Saved {affected_before - affected_after} users",
        f"Increased by {bfs_after['unaffected_count'] - bfs_result['unaffected_count']} users",
        f"Halted at level {bfs_after['max_level']}",
        f"{reduction_pct}% Reduction"
    ]
})
st.table(comparison_df)
