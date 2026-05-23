import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import plotly.subplots as ps
import plotly.graph_objects as go

import config
from data_loader import load_data, load_tickers
from correlation import get_correlation_matrices


def plot_clustered_heatmap(corr_matrix, tickers, title="", cmap='YlOrRd'):
    n = len(tickers)
    dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)

    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method='ward')
    order = leaves_list(linkage_matrix)

    corr_ordered = corr_matrix[order][:, order]
    tickers_ordered = [tickers[i] for i in order]

    split_idx = n // 2

    plt.figure(figsize=(10, 8))
    im = plt.imshow(np.abs(corr_ordered), cmap=cmap, vmin=0, vmax=1)
    plt.axhline(split_idx - 0.5, linestyle='--')
    plt.axvline(split_idx - 0.5, linestyle='--')
    plt.title(title)
    plt.colorbar(im, label='|Correlation|')

    if n <= 50:
        plt.xticks(range(n), tickers_ordered, rotation=90)
        plt.yticks(range(n), tickers_ordered)

    plt.tight_layout()
    plt.show()
    return order, split_idx


def analyze_correlation_network(corr_matrix, tickers, title="", edge_threshold=0.2, top_n_nodes=100, centrality_metric='pagerank'):
    n = corr_matrix.shape[0]
    G = nx.DiGraph()

    for ticker in tickers:
        G.add_node(ticker)

    for i in range(n):
        for j in range(n):
            if i != j:
                weight = corr_matrix[i, j]
                if not np.isnan(weight):
                    abs_weight = abs(weight)
                    if abs_weight >= edge_threshold:
                        G.add_edge(tickers[i], tickers[j], weight=weight, abs_weight=abs_weight)

    if centrality_metric == 'pagerank':
        centrality = nx.pagerank(G, weight='abs_weight')
    elif centrality_metric == 'out_strength':
        centrality = {node: sum(data['abs_weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}
    elif centrality_metric == 'betweenness':
        centrality = nx.betweenness_centrality(G, weight='abs_weight')
    else:
        centrality = {node: G.out_degree(node, weight='abs_weight') for node in G.nodes()}

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    display_nodes = [ticker for ticker, _ in sorted_nodes[:top_n_nodes]]
    H = G.subgraph(display_nodes).copy()

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(H, k=2, iterations=100, seed=42)

    node_sizes = [min(300 + centrality.get(ticker, 0) * 1500, 2500) for ticker in H.nodes()]
    node_centralities = [centrality.get(ticker, 0) for ticker in H.nodes()]

    edge_colors = ['red' if H[u][v]['weight'] < 0 else 'green' for u, v in H.edges()]
    edge_widths = [min(H[u][v]['abs_weight'] * 6, 5) for u, v in H.edges()]

    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6, arrows=True, arrowsize=15, arrowstyle='->')
    nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_centralities, cmap='viridis', alpha=0.9, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(H, pos, font_size=9, font_weight='bold')

    plt.colorbar(nodes, shrink=0.8, label=f'{centrality_metric} Score')
    plt.title(f'{title}\n', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return G, centrality


def analyze_correlation_network_v2(corr_matrix, tickers, title="", edge_threshold=0.2, top_n_nodes=50):
    n = corr_matrix.shape[0]
    G = nx.DiGraph()

    for ticker in tickers:
        G.add_node(ticker)

    for i in range(n):
        for j in range(n):
            if i != j:
                weight = corr_matrix[i, j]
                if not np.isnan(weight):
                    abs_weight = abs(weight)
                    if abs_weight >= edge_threshold:
                        G.add_edge(tickers[i], tickers[j], weight=weight, abs_weight=abs_weight)

    out_strength = {node: sum(data['abs_weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}
    sorted_nodes = sorted(out_strength.items(), key=lambda x: x[1], reverse=True)
    display_nodes = [node for node, _ in sorted_nodes[:top_n_nodes]]

    H = G.subgraph(display_nodes).copy()
    pos = {}
    sorted_display = sorted(H.nodes(), key=lambda x: out_strength[x], reverse=True)

    for idx, node in enumerate(sorted_display):
        pos[node] = (-out_strength[node], idx)

    node_sizes = [300 + out_strength[node] * 1000 for node in H.nodes()]
    node_colors = [out_strength[node] for node in H.nodes()]
    edge_widths = [H[u][v]['abs_weight'] * 8 for u, v in H.edges()]
    edge_colors = ['darkred' if H[u][v]['weight'] < 0 else 'darkgreen' for u, v in H.edges()]

    plt.figure(figsize=(10, 12))
    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, width=edge_widths, alpha=0.5, arrows=True, arrowsize=20)
    nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, cmap='plasma', edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(H, pos, font_size=9, font_weight='bold')

    plt.colorbar(nodes, label='Out-strength')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return G, out_strength


def analyze_patterns(corr_ol_dt, corr_dt_ol):
    ol_corrs = corr_ol_dt[corr_ol_dt != 0].flatten()
    dt_corrs = corr_dt_ol[corr_dt_ol != 0].flatten()

    fig = ps.make_subplots(rows=1, cols=2, subplot_titles=["Overnight-Lead-Daytime", "Daytime-Lead-Overnight"])
    fig.add_trace(go.Histogram(x=ol_corrs, name="OL→DT", nbinsx=30), row=1, col=1)
    fig.add_trace(go.Histogram(x=dt_corrs, name="DT→OL", nbinsx=30), row=1, col=2)
    fig.update_layout(height=400, title_text="Correlation Distributions")
    fig.show()


def main():
    tickers, meta_df = load_tickers() 
    abs_corr_ol_dt, abs_corr_dt_ol = get_correlation_matrices(tickers, lookback_days=60)

    analyze_patterns(abs_corr_ol_dt, abs_corr_dt_ol)

    order, split_idx = plot_clustered_heatmap(abs_corr_ol_dt[:50, :50], tickers[:50], title="Overnight → Daytime")

    G, centrality = analyze_correlation_network(
        corr_matrix=abs_corr_ol_dt, tickers=tickers, title="Correlation overnight -> daytime",
        edge_threshold=0.45, top_n_nodes=50, centrality_metric='out_strength'
    )

    analyze_correlation_network_v2(
        corr_matrix=abs_corr_ol_dt, tickers=tickers, title="Correlation Overnight → Daytime",
        edge_threshold=0.45, top_n_nodes=20
    )

    # Эго-сеть влияния для EQIX 
    leader = 'EQIX'
    neighbors = list(G.successors(leader))
    sub_nodes = [leader] + neighbors

    H = G.subgraph(sub_nodes)
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(H, seed=42)
    edge_weights = [H[u][v]['abs_weight'] * 10 for u, v in H.edges()]

    nx.draw_networkx(H, pos, with_labels=True, node_size=2000, node_color='lightblue', width=edge_weights, arrows=True, font_weight='bold')
    plt.title(f"Сеть влияния {leader} на акции")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()