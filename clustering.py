import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from data_loader import load_tickers
from correlation import get_correlation_matrices


def get_cluster_hierarchy(labels, A_abs, ticker_names=None):
    n_clusters = len(np.unique(labels))
    clusters = {k: np.where(labels == k)[0] for k in range(n_clusters)}

    # Строим матрицу средних потоков между кластерами
    flow_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j and len(clusters[i]) > 0 and len(clusters[j]) > 0:
                flow_matrix[i, j] = np.mean(A_abs[np.ix_(clusters[i], clusters[j])])

    # Считаем lead_score для каждого кластера
    lead_scores = {}
    for i in range(n_clusters):
        out_flows = [flow_matrix[i, j] for j in range(n_clusters) if i != j]
        lead_scores[i] = np.mean(out_flows) if out_flows else 0

    sorted_clusters = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)

    hierarchy = []
    for cluster_idx, score in sorted_clusters:
        if ticker_names is not None:
            cluster_tickers = [ticker_names[i] for i in clusters[cluster_idx]]
        else:
            cluster_tickers = clusters[cluster_idx]
            
        hierarchy.append({
            'cluster_id': cluster_idx,
            'tickers': cluster_tickers,
            'lead_score': score,
            'size': len(clusters[cluster_idx])
        })

    main_leader = hierarchy[0] if hierarchy else None
    main_lagger = hierarchy[-1] if len(hierarchy) > 1 else None

    leader_to_lagger_score = 0
    if main_leader and main_lagger:
        leader_to_lagger_score = flow_matrix[main_leader['cluster_id'], main_lagger['cluster_id']]

    return {
        'hierarchy': hierarchy,
        'main_leader': main_leader,
        'main_lagger': main_lagger,
        'leader_to_lagger_score': leader_to_lagger_score,
        'flow_matrix': flow_matrix
    }


class DirectedLeadLagClusterer:
    def __init__(self, num_iterations=10, initial_eta=None, n_clusters=2):
        self.num_iterations = num_iterations
        self.initial_eta = initial_eta if initial_eta is not None else np.random.uniform(0.1, 0.4)
        self.n_clusters = n_clusters  
        self.eta_history = []
        self.flow_history = []

    def _compute_H(self, A, eta):
        safe_eta = np.clip(eta, 1e-10, 0.5 - 1e-10)
        w_i = np.log((1 - safe_eta) / safe_eta)
        w_r = np.log(1 / (4 * safe_eta * (1 - safe_eta)))
        H = 1j * w_i * (A - A.T) + w_r * (A + A.T)
        return H

    def _compute_flow(self, A, source, target):
        return np.sum(A[np.ix_(source, target)])

    def fit(self, correlation_matrix, ticker_names=None):
        self.ticker_names = ticker_names
        self.corr_matrix = correlation_matrix
        A = np.abs(correlation_matrix)
        self.A_abs = A
        eta = self.initial_eta

        for i in range(self.num_iterations):
            H = self._compute_H(A, eta)
            eigvals, eigvecs = np.linalg.eig(H)

            top_indices = np.argsort(np.real(eigvals))[-self.n_clusters+1:]
            embedding = []
            for idx in top_indices:
                v = eigvecs[:, idx]
                embedding.append(np.real(v))
                embedding.append(np.imag(v))

            embedding = np.column_stack(embedding)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42 + i, n_init=10)
            self.labels_ = kmeans.fit_predict(embedding)  # Сохраняем labels_ для совместимости

            self.clusters_ = {}
            for k in range(self.n_clusters):
                self.clusters_[k] = np.where(self.labels_ == k)[0]

            total_flow = 0
            min_flow_ratio = 1

            for c1 in range(self.n_clusters):
                for c2 in range(self.n_clusters):
                    if c1 != c2:
                        flow = self._compute_flow(A, self.clusters_[c1], self.clusters_[c2])
                        total_flow += flow
                        flow_ratio = flow / (self._compute_flow(A, self.clusters_[c2], self.clusters_[c1]) + 1e-10)
                        min_flow_ratio = min(min_flow_ratio, flow_ratio)

            if total_flow > 0:
                eta_new = min(min_flow_ratio, 0.5)
            else:
                eta_new = eta

            if np.abs(eta_new - eta) < 1e-6:
                break
            eta = eta_new

        return self


class HermitianClusterer:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, adjacency_matrix):
        A = np.asarray(adjacency_matrix)
        H = 1j * (A - A.T) + (A + A.T)

        _, eigvecs = np.linalg.eigh(H)
        embedding = eigvecs[:, :self.n_clusters]

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = kmeans.fit_predict(np.real(embedding))
        return self


class BibliometricClusterer:
    def __init__(self, n_clusters=2, method='coupling'):
        self.n_clusters = n_clusters
        self.method = method

    def fit(self, adjacency_matrix):
        A = np.asarray(adjacency_matrix)

        if self.method == 'coupling':
            S = A @ A.T
        elif self.method == 'cocitation':
            S = A.T @ A
        else:
            raise ValueError(f"нет метода '{self.method}'")

        clustering = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=42)
        self.labels_ = clustering.fit_predict(S)
        return self


class SVDClusterer:
    def __init__(self, n_clusters=2, n_components=None):
        self.n_clusters = n_clusters
        self.n_components = n_components

    def fit(self, adjacency_matrix):
        A = np.asarray(adjacency_matrix)
        n_comp = self.n_components if self.n_components else self.n_clusters
        
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        embedding = svd.fit_transform(A)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = kmeans.fit_predict(embedding)
        return self


def evaluate_clusterer(name, clusterer, matrix, tickers):
    clusterer.fit(matrix)
    
    # Универсальное извлечение labels_
    if hasattr(clusterer, 'labels_'):
        labels = clusterer.labels_
    else:
        raise AttributeError(f"{name} не имеет атрибута labels_ после обучения.")

    hierarchy_info = get_cluster_hierarchy(labels, matrix, tickers)
    
    for i, clust in enumerate(hierarchy_info['hierarchy']):
        status = "ЛИДЕР" if i == 0 else "ЛАГГЕР" if i == len(hierarchy_info['hierarchy'])-1 else ""
        print(f"  Кластер {clust['cluster_id']} {status}:")
        print(f"    Score: {clust['lead_score']:.4f}")
        print(f"    Размер: {clust['size']} акций")
        print(f"    Примеры: {clust['tickers'][:5]}")
        print("-" * 30)

    print(f"Главный лидер → Главный лаггер (поток): {hierarchy_info['leader_to_lagger_score']:.4f}\n")
    return hierarchy_info


def main():
    tickers, _ = load_tickers()
    
    abs_corr_ol_dt, abs_corr_dt_ol = get_correlation_matrices(tickers, lookback_days=60)
    matrix_to_cluster = abs_corr_ol_dt  # Для кластеризации используем абсолютные значения

    n_clusters = 4

    # Directed Lead-Lag
    dll_clusterer = DirectedLeadLagClusterer(num_iterations=50, n_clusters=n_clusters)
    hierarchy_dll = evaluate_clusterer("Directed Lead-Lag", dll_clusterer, matrix_to_cluster, tickers)
    leaders = hierarchy_dll['main_leader']['tickers']
    laggers = hierarchy_dll['main_lagger']['tickers']

    # Hermitian
    herm_clusterer = HermitianClusterer(n_clusters=n_clusters)
    hierarchy_herm = evaluate_clusterer("Hermitian Complex Embedding", herm_clusterer, matrix_to_cluster, tickers)
    leaders_herm = hierarchy_herm['main_leader']['tickers']
    laggers_herm = hierarchy_herm['main_lagger']['tickers']

    # Bibliometric
    bibl_clusterer = BibliometricClusterer(n_clusters=n_clusters, method='coupling')
    hierarchy_bibl = evaluate_clusterer("Bibliometric Symmetrization (Coupling)", bibl_clusterer, matrix_to_cluster, tickers)
    leaders_bibl = hierarchy_bibl['main_leader']['tickers']
    laggers_bibl = hierarchy_bibl['main_lagger']['tickers']

    # SVD
    svd_clusterer = SVDClusterer(n_clusters=n_clusters)
    hierarchy_svd = evaluate_clusterer("Truncated SVD + KMeans", svd_clusterer, matrix_to_cluster, tickers)
    leaders_svd = hierarchy_svd['main_leader']['tickers']
    laggers_svd = hierarchy_svd['main_lagger']['tickers']
    return (
        leaders, laggers,
        leaders_herm, laggers_herm,
        leaders_bibl,laggers_bibl,
       leaders_svd, laggers_svd)
    


if __name__ == "__main__":
    results = main()
