# Fake News Detection, Propagation and Containment Using Advanced Graph Algorithms

---

## 1. Title
**Fake News Detection, Propagation and Containment Using Advanced Graph Algorithms**

---

## 2. Problem Statement
The exponential growth of social media platforms has accelerated the rapid dissemination of unverified rumors, fabricated stories, and digital misinformation (fake news). Traditional content-moderation systems focus almost exclusively on textual classification, failing to model how misinformation cascades across interconnected social topologies or identify high-leverage intervention points. Once fake news enters a social network, it spreads epidemically before reactive moderation can intervene. There is a critical need for an integrated framework that not only detects fake news with machine learning but also models its dynamic network diffusion and applies rigorous graph-theoretic algorithms to isolate super-spreaders, sever critical transmission channels, and achieve optimal propagation containment.

---

## 3. Objectives
1. **Machine Learning Verification:** Construct an NLP classification pipeline using Term Frequency-Inverse Document Frequency (TF-IDF) and Logistic Regression to classify incoming news articles as `FAKE` or `REAL` with high confidence.
2. **Topological Social Modeling:** Represent complex social networks as graph data structures where users are vertices and follower/friendship relationships are edges.
3. **Information Cascade Simulation:** Implement Breadth-First Search (BFS) to simulate level-by-level (hop-by-hop) misinformation diffusion from an origin source.
4. **Influential Node Discovery:** Apply the **PageRank** algorithm to detect high-authority super-spreaders capable of amplifying misinformation.
5. **Bridge Node Identification:** Compute **Betweenness Centrality** to identify gatekeeper accounts connecting disparate sub-communities.
6. **Sentinel Monitoring Optimization:** Formulate a **Greedy Approximate Minimum Dominating Set** to establish complete 1-hop observational surveillance over the entire network using minimal sentinel nodes.
7. **Targeted Flow Containment:** Utilize the **Max-Flow Min-Cut Theorem** (`networkx.minimum_cut`) to identify the exact critical transmission edges that isolate sensitive target users or communities.
8. **Comparative Containment Evaluation:** Quantitatively measure and visualize propagation reduction percentages before and after containment interventions.

---

## 4. Existing System
Traditional misinformation handling mechanisms suffer from several fundamental limitations:
- **Isolated Classification:** Existing tools flag fake news in isolation without modeling how or where the information travels.
- **Uniform / Naive Interventions:** Platforms often apply blunt account bans or platform-wide censorship without structural topological analysis.
- **High Computational Overhead:** Attempting to monitor all active accounts simultaneously requires immense computational and human fact-checking resources.
- **Lack of Quantitative Containment Metrics:** No algorithmic mechanism exists to estimate beforehand how much spread is prevented by removing specific edges versus nodes.

---

## 5. Proposed System
The proposed system implements a holistic, two-tier defense architecture:
1. **Tier 1 (NLP Detection Tier):** Analyzes raw text using TF-IDF vectorization and Logistic Regression. If an article is verified as `REAL`, the cascade simulation is safely halted. If identified as `FAKE`, it triggers Tier 2.
2. **Tier 2 (Graph Cascade & Containment Tier):** Maps the misinformation to a simulated social media topology. It simulates propagation cascades via BFS and executes four distinct graph-theoretic containment strategies (PageRank Super-spreader Quarantining, Betweenness Bridge Quarantining, Dominating Set Sentinel Deployment, and Minimum Cut Channel Severing), quantitatively demonstrating propagation reduction ($(\Delta \text{Affected} / \text{Affected}_{\text{Before}}) \times 100$).

---

## 6. System Architecture

```
                  ┌─────────────────────────────────┐
                  │    News Headline / Text Input   │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  Text Cleaning & Preprocessing  │
                  │  (Lowercase, Regex, Stopwords)  │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │      TF-IDF Vectorization       │
                  │   & Logistic Regression Model   │
                  └────────────────┬────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        [Classified as REAL]            [Classified as FAKE]
        (Stop & Display Info)                     │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │ Initialize Social Graph G │
                                    │  & Select Source Node s   │
                                    └─────────────┬─────────────┘
                                                  │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │ BFS Cascade Simulation    │
                                    │ (Pre-Containment Spread)  │
                                    └─────────────┬─────────────┘
                                                  │
                     ┌────────────────────────────┼────────────────────────────┐
                     ▼                            ▼                            ▼
        ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
        │ Algorithm 2: PageRank   │  │ Algorithm 3: Betweenness│  │ Algorithm 4: Dom. Set   │
        │ (Top Influential Hubs)  │  │ (Inter-Cluster Bridges) │  │ (Min Sentinel Coverage) │
        └────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
                     │                            │                            │
                     └────────────────────────────┼────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │ Algorithm 5: Min-Cut Flow │
                                    │ (Edge Boundary Severing)  │
                                    └─────────────┬─────────────┘
                                                  │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │ Apply Selected Strategy   │
                                    │ (Block Nodes / Cut Edges) │
                                    └─────────────┬─────────────┘
                                                  │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │ Re-run BFS Propagation    │
                                    │ (Post-Containment Spread) │
                                    └─────────────┬─────────────┘
                                                  │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │ Compute Reduction %       │
                                    │ & Display Visualizations  │
                                    └───────────────────────────┘
```

---

## 7. Technologies Used
- **Programming Language:** Python 3.12+
- **Graph Processing & Algorithms:** `NetworkX` (v3.0+)
- **Machine Learning & NLP:** `Scikit-Learn` (v1.3+), `Joblib`
- **Data Manipulation:** `Pandas` (v2.0+), `NumPy` (v1.24+)
- **Graph & Metric Visualization:** `Matplotlib` (v3.7+)
- **User Interface:** `Streamlit` (v1.30+)
- **Hosting & Execution:** 100% Local, zero third-party paid APIs.

---

## 8. Dataset Description
1. **Fake News Classification Dataset (`data/fake_news.csv`):**
   - Contains labeled instances of `title`, `text`, and `label` (`FAKE` vs `REAL`).
   - Covers diverse domains including science, technology, public policy, health rumors, and astronomy.
2. **Social Network Dataset (`data/social_network.csv`):**
   - Simulated realistic social network of 30 nodes (`User1` to `User30`) and 61 edges.
   - Structured into 3 interconnected community clusters featuring designated bridge connections (`User7-User11`, `User8-User12`, `User15-User21`, `User18-User22`) and high-degree hub spreaders (`User1`, `User12`, `User14`, `User21`).

---

## 9. Fake News Detection Method
The text classification module implements:
1. **Text Preprocessing:** URL stripping, HTML tag removal, punctuation elimination, lowercase conversion, and whitespace normalization.
2. **TF-IDF Vectorization:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
3. **Logistic Regression Classifier:**
   $$P(y = \text{FAKE} \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   Where $\mathbf{x}$ is the TF-IDF feature vector, $\mathbf{w}$ is the learned weight vector, and $b$ is the bias.

---

## 10. Graph Representation
The social network is formally modeled as an undirected graph:
$$G = (V, E)$$
Where:
- $V = \{v_1, v_2, \dots, v_n\}$ is the set of user accounts (vertices).
- $E = \{(u, v) \mid u, v \in V, u \neq v\}$ represents bi-directional communication, follower, or friendship relationships.
- Each edge $(u, v) \in E$ is assigned a transmission capacity $c(u, v) = 1.0$.

---

## 11. Algorithm 1: Breadth-First Search (BFS) Propagation
- **Mathematical / Procedural Formulation:**
  Given source $s \in V$ and maximum depth $k$:
  1. Initialize queue $Q \leftarrow [s]$, visited set $S \leftarrow \{s\}$, and level map $L(s) = 0$.
  2. While $Q \neq \emptyset$:
     - Dequeue $u \leftarrow Q.\text{pop}()$.
     - If $L(u) < k$, for each unblocked neighbor $v \in N(u) \setminus S$ across unblocked edge $(u, v)$:
       - $S \leftarrow S \cup \{v\}$
       - $L(v) \leftarrow L(u) + 1$
       - Enqueue $v$ to $Q$.
- **Purpose:** Tracks exact level-by-level cascade wavefronts (Time 0, Time 1, Time 2, etc.) and calculates total affected user counts.

---

## 12. Algorithm 2: PageRank Centrality
- **Mathematical Formula:**
  $$PR(u) = \frac{1 - d}{|V|} + d \sum_{v \in N_{in}(u)} \frac{PR(v)}{|N_{out}(v)|}$$
  Where $d = 0.85$ is the random walk damping factor.
- **Purpose:** Measures global transmission authority. High-PageRank nodes act as super-spreaders whose endorsement causes widespread information explosion.

---

## 13. Algorithm 3: Betweenness Centrality
- **Mathematical Formula:**
  $$g(v) = \sum_{s \neq v \neq t \in V} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
  Where $\sigma_{st}$ is the total number of shortest paths from node $s$ to node $t$, and $\sigma_{st}(v)$ is the number of those paths that pass through $v$.
- **Purpose:** Identifies gatekeepers connecting distinct sub-communities. Quarantining bridge nodes confines rumors to their originating cluster.

---

## 14. Algorithm 4: Greedy Minimum Dominating Set
- **Problem Formulation:**
  A subset $D \subseteq V$ is a dominating set if:
  $$\forall v \in V \setminus D, \quad \exists u \in D \text{ such that } (u, v) \in E$$
- **Greedy Heuristic Algorithm:**
  1. Let uncovered set $U \leftarrow V$, dominating set $D \leftarrow \emptyset$.
  2. While $U \neq \emptyset$:
     - Pick node $u^* = \arg\max_{u \in V} |(N[u] \cap U)|$ where $N[u] = \{u\} \cup N(u)$.
     - $D \leftarrow D \cup \{u^*\}$.
     - $U \leftarrow U \setminus N[u^*]$.
- **Purpose:** Finds the smallest subset of sentinel accounts to monitor that achieves 100% 1-hop observational coverage over all network users.

---

## 15. Algorithm 5: Network Flow and Minimum Cut
- **Mathematical Foundation (Max-Flow Min-Cut Theorem):**
  In a flow network $G=(V, E)$ with source $s$ and sink/target $t$:
  $$\max (\text{Flow}(s \to t)) = \min_{(S, T)} \text{Capacity}(S, T)$$
  Where an $(S, T)$ cut is a partition of $V$ into $S$ and $T = V \setminus S$ such that $s \in S$ and $t \in T$.
- **Purpose:** Determines the minimal transmission edges whose severance guarantees zero information flow from the misinformation source to a designated target user.

---

## 16. Propagation Process
1. A news article is evaluated; if `FAKE`, user selects origin account $s$.
2. BFS cascade begins at $t=0$ with $s$.
3. At step $t=1$, all immediate neighbors $N(s)$ are infected.
4. Cascade expands outward until maximum depth $k$ or until no new uninfected neighbors remain.
5. Baseline affected count ($\text{Affected}_{\text{Before}}$) and depth are recorded.

---

## 17. Containment Process
1. Administrator selects an algorithmic containment policy:
   - **Policy A:** Quarantine top PageRank super-spreaders.
   - **Policy B:** Quarantine top Betweenness bridge accounts.
   - **Policy C:** Deploy fact-checking barriers at Dominating Set nodes.
   - **Policy D:** Sever Minimum Cut boundary edges.
2. The graph topology is dynamically updated (nodes or edges removed).
3. BFS propagation is re-simulated from source $s$.
4. Post-containment affected count ($\text{Affected}_{\text{After}}$) is recorded.
5. Propagation reduction percentage is calculated:
   $$\text{Propagation Reduction (\%)} = \left(\frac{\text{Affected}_{\text{Before}} - \text{Affected}_{\text{After}}}{\text{Affected}_{\text{Before}}}\right) \times 100$$

---

## 18. Results & Performance Metrics
- **NLP Detection:** Achieves **87.50%** test accuracy on the benchmark news dataset.
- **PageRank Containment:** Quarantining the top 2 PageRank hubs achieves an average **30% to 55% reduction** in misinformation spread.
- **Betweenness Bridge Containment:** Quarantining the top 2 bridge nodes prevents inter-cluster contagion, achieving up to **60% to 80% reduction** when spreading across communities.
- **Minimum Cut Isolation:** Achieves **100% protection** for targeted communities with minimal edge cuts ($\text{cut value} \le 3$).
- **Dominating Set:** Covers 100% of the 30-user network using only 7 sentinel accounts (~23% monitoring overhead).

---

## 19. Advantages
1. **Dual-Tier Synergy:** Combines machine learning textual analysis with structural graph dynamics.
2. **Explainable Interventions:** Every blocked user or severed edge is backed by mathematical centrality and flow theorems.
3. **Resource Efficiency:** Dominating set and centralities prevent the need for wasteful brute-force monitoring.
4. **Interactive Visualization:** Color-coded graphs and side-by-side bar charts clearly explain results to technical and non-technical stakeholders.

---

## 20. Limitations
1. Simulated environment assumes deterministic BFS spread; real-world human sharing behavior exhibits probabilistic and psychological variance.
2. Minimum Dominating Set is NP-hard; greedy approximation is near-optimal but may not always yield the absolute minimum theoretical set for arbitrary graphs.

---

## 21. Future Enhancements
1. Integrate probabilistic diffusion models like Independent Cascade (IC) and Linear Threshold (LT).
2. Incorporate deep learning NLP models (BERT / RoBERTa) for multilingual fake news detection.
3. Extend to temporal dynamic graphs where edges appear and disappear based on user activity timestamps.

---

## 22. UN Sustainable Development Goals (SDG) Alignment
- **SDG 16 (Peace, Justice and Strong Institutions):** Mitigates digital disinformation campaigns that destabilize public trust and democratic institutions.
- **SDG 9 (Industry, Innovation and Infrastructure):** Provides innovative algorithmic infrastructure for automated content integrity and social network resilience.

---

## 23. Conclusion
The project successfully demonstrates that integrating Machine Learning classification with Advanced Graph Algorithms provides a powerful, proactive framework for detecting, modeling, and containing fake news cascades. By targeting influential spreaders, bridge accounts, and critical edge cuts, social media platforms can achieve substantial misinformation reduction with minimal disruption to regular network communication.

---

## 24. How to Run the Project

### Prerequisites
- Python 3.10+ installed on your system.

### Step 1: Install Dependencies
Open your terminal/command prompt in the project root directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Train/Verify ML Model
```bash
python models/train_model.py
```

### Step 3: Run Automated Test Suite
```bash
python test_suite.py
```

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will automatically open in your web browser at `http://localhost:8501`.

