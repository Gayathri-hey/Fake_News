"""
Automated Verification and Unit Test Suite for:
Fake News Detection, Propagation, and Containment Project.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.preprocessing import clean_text, train_fake_news_classifier, predict_news, load_model
from utils.graph_utils import generate_default_social_network, get_graph_metrics, validate_source_target
from algorithms.propagation import simulate_bfs_propagation
from algorithms.pagerank import compute_pagerank
from algorithms.centrality import compute_betweenness_centrality
from algorithms.dominating_set import compute_approx_dominating_set
from algorithms.network_flow import compute_minimum_cut_containment
from utils.visualization import (
    get_fixed_layout,
    plot_social_network,
    plot_propagation_graph,
    plot_highlighted_nodes,
    plot_containment_network,
    plot_comparison_bar_chart
)
import matplotlib.pyplot as plt


class TestFakeNewsProject(unittest.TestCase):
    
    def setUp(self):
        self.G = generate_default_social_network()
        self.data_path = os.path.join(BASE_DIR, "data", "fake_news.csv")
        self.model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
        
    def test_text_cleaning(self):
        raw = "Check this out! https://fake.url <b>Secret</b> 100% cure."
        cleaned = clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("secret", cleaned)
        
    def test_ml_model_prediction(self):
        model = load_model(self.model_path)
        self.assertIsNotNone(model, "Model checkpoint fake_news_model.pkl should exist.")
        
        pred, conf, probs = predict_news(model, "Government secretly installs 5G chips in salt")
        self.assertIn(pred, ['FAKE', 'REAL'])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        
    def test_graph_metrics(self):
        metrics = get_graph_metrics(self.G)
        self.assertEqual(metrics['num_nodes'], 30)
        self.assertGreater(metrics['num_edges'], 50)
        self.assertTrue(metrics['is_connected'])
        
    def test_bfs_propagation(self):
        res = simulate_bfs_propagation(self.G, source="User1", max_steps=4)
        self.assertTrue(res['success'])
        self.assertGreater(res['affected_count'], 1)
        self.assertEqual(res['source'], "User1")
        self.assertIn("User1", res['affected_users'])
        
    def test_pagerank(self):
        pr_res = compute_pagerank(self.G, top_k=5)
        self.assertEqual(len(pr_res['top_users']), 5)
        self.assertEqual(len(pr_res['scores']), 30)
        self.assertAlmostEqual(sum(pr_res['scores'].values()), 1.0, places=3)
        
    def test_betweenness_centrality(self):
        bc_res = compute_betweenness_centrality(self.G, top_k=5)
        self.assertEqual(len(bc_res['top_users']), 5)
        self.assertEqual(len(bc_res['scores']), 30)
        
    def test_dominating_set(self):
        dom_res = compute_approx_dominating_set(self.G)
        self.assertGreater(dom_res['selected_count'], 0)
        self.assertLess(dom_res['selected_count'], 30)
        self.assertEqual(dom_res['coverage_percentage'], 100.0)
        
    def test_minimum_cut(self):
        min_cut = compute_minimum_cut_containment(self.G, "User1", "User30")
        self.assertTrue(min_cut['success'])
        self.assertGreater(min_cut['cut_value'], 0)
        self.assertGreater(len(min_cut['critical_edges']), 0)
        
    def test_containment_reduction(self):
        # BFS Before
        before = simulate_bfs_propagation(self.G, source="User1", max_steps=5)
        
        # Containment via blocking top 2 PageRank nodes
        pr_res = compute_pagerank(self.G, top_k=2)
        after = simulate_bfs_propagation(
            self.G, source="User1", max_steps=5, blocked_nodes=pr_res['top_users']
        )
        
        self.assertLessEqual(after['affected_count'], before['affected_count'])
        reduction = ((before['affected_count'] - after['affected_count']) / before['affected_count']) * 100
        self.assertGreaterEqual(reduction, 0.0)
        
    def test_visualizations(self):
        pos = get_fixed_layout(self.G)
        fig1, _ = plot_social_network(self.G, pos=pos)
        self.assertIsNotNone(fig1)
        plt.close(fig1)
        
        bfs = simulate_bfs_propagation(self.G, source="User1", max_steps=3)
        fig2 = plot_propagation_graph(self.G, pos=pos, source="User1", levels=bfs['levels'])
        self.assertIsNotNone(fig2)
        plt.close(fig2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
