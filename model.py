import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

DATA_PATH = Path(__file__).parent / "data" / "products.json"


ACTION_WEIGHT = {
    "view": 1.0, 
    "click": 2.0,
    "cart": 3.5, 
}

RECENCY_HALF_LIFE = 4 


class HybridRecommender:
    def __init__(self):
        self.products = json.loads(DATA_PATH.read_text())
        self.id_to_idx = {p["id"]: i for i, p in enumerate(self.products)}
        self.n_items = len(self.products)

        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self):
        corpus = [
            f"{p['name']} {p['brand']} {p['category']} "
            f"{' '.join(p['tags'])} {p['description']}"
            for p in self.products
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.content_sim = cosine_similarity(tfidf_matrix)

    def _build_collaborative_model(self, n_synthetic_users=400, n_components=10, seed=42):
        rng = np.random.default_rng(seed)
        categories = sorted({p["category"] for p in self.products})
        cat_items = {c: [i for i, p in enumerate(self.products) if p["category"] == c] for c in categories}

        affinity = {
            "Synthesizers": ["Drum Machines", "Cables & Accessories"],
            "Drum Machines": ["Synthesizers", "Effects Pedals"],
            "Effects Pedals": ["Drum Machines", "Cables & Accessories"],
            "Eurorack Modules": ["Cables & Accessories", "Synthesizers"],
            "Cables & Accessories": ["Eurorack Modules"],
        }

        R = np.zeros((n_synthetic_users, self.n_items))
        for u in range(n_synthetic_users):
            primary = rng.choice(categories)
            secondary = rng.choice(affinity.get(primary, categories))
            for i in cat_items[primary]:
                R[u, i] += rng.poisson(3.0)
            for i in cat_items[secondary]:
                R[u, i] += rng.poisson(1.2)
            # light cross-shopping noise so the matrix isn't block-diagonal
            noise_items = rng.choice(self.n_items, size=3, replace=False)
            for i in noise_items:
                R[u, i] += rng.poisson(0.4)

        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.user_factors = self.svd.fit_transform(R)       
        self.item_factors = self.svd.components_.T          

        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self._item_factors_unit = self.item_factors / norms

    def _fold_in_session(self, events):
        if not events:
            return None
        vec = np.zeros(self.item_factors.shape[1])
        total_w = 0.0
        n = len(events)
        for rank, ev in enumerate(reversed(events)): 
            idx = self.id_to_idx.get(ev["product_id"])
            if idx is None:
                continue
            recency_decay = 0.5 ** (rank / RECENCY_HALF_LIFE)
            w = ACTION_WEIGHT.get(ev["action"], 1.0) * recency_decay
            vec += w * self.item_factors[idx]
            total_w += w
        if total_w == 0:
            return None
        vec /= total_w
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm

    def _content_score(self, events):
        if not events:
            return None
        scores = np.zeros(self.n_items)
        total_w = 0.0
        for rank, ev in enumerate(reversed(events)):
            idx = self.id_to_idx.get(ev["product_id"])
            if idx is None:
                continue
            recency_decay = 0.5 ** (rank / RECENCY_HALF_LIFE)
            w = ACTION_WEIGHT.get(ev["action"], 1.0) * recency_decay
            scores += w * self.content_sim[idx]
            total_w += w
        if total_w == 0:
            return None
        return scores / total_w

    def _dynamic_alpha(self, events):
        
        if not events:
            return 0.7
        last_action = events[-1]["action"]
        depth_penalty = min(len(events), 6) * 0.05
        alpha = 0.75 - depth_penalty
        if last_action == "cart":
            alpha -= 0.2
        return max(0.25, min(0.8, alpha))

    def recommend(self, events, cart_ids=None, top_n=6):
       
        cart_ids = set(cart_ids or [])
        recent_ids = {ev["product_id"] for ev in events[-3:]} if events else set()
        exclude = cart_ids | recent_ids

        content = self._content_score(events)
        cf_vec = self._fold_in_session(events)
        cf = None
        if cf_vec is not None:
            cf = self._item_factors_unit @ cf_vec

        if content is None and cf is None:
            ranked = list(range(self.n_items))
            ranked.sort(key=lambda i: -self.products[i]["price"] % 7)
            reason = "popular"
        else:
            alpha = self._dynamic_alpha(events)
            if content is None:
                blended = cf
                reason = "collaborative"
            elif cf is None:
                blended = content
                reason = "content"
            else:
                blended = alpha * content + (1 - alpha) * cf
                reason = "hybrid"
            ranked = list(np.argsort(-blended))

        results = []
        for idx in ranked:
            pid = self.products[idx]["id"]
            if pid in exclude:
                continue
            item = dict(self.products[idx])
            item["_reason"] = reason
            results.append(item)
            if len(results) >= top_n:
                break
        return results, (self._dynamic_alpha(events) if events else None)


engine = HybridRecommender()
