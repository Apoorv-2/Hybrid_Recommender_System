import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "products.json"

ACTION_WEIGHT = {
    "view": 1.0,
    "click": 2.0,
    "cart": 3.5,
}

RECENCY_HALF_LIFE = 4

CO_PURCHASE_PAIRS = [

    (1, 26, 2.5), (1, 27, 1.5), (2, 26, 2.0), (2, 27, 1.2), (3, 27, 1.5),
    (4, 1, 2.0), (4, 2, 1.5), (5, 2, 1.5), (5, 3, 1.2),
    (11, 21, 3.0), (12, 22, 1.5), (13, 23, 1.0), (14, 22, 1.5),
    (16, 22, 1.2), (17, 23, 1.0), (18, 22, 1.0),
    (11, 92, 2.0), (12, 91, 1.5), (13, 91, 1.5), (14, 91, 1.5), (17, 91, 1.2),
    (11, 72, 2.5), (12, 71, 2.0), (13, 69, 1.0), (14, 70, 1.0), (16, 69, 1.0),
    (85, 80, 1.5), (85, 81, 1.2), (85, 88, 1.5), (85, 94, 1.0),
    (86, 72, 1.5), (86, 92, 1.5), (86, 95, 1.8), (86, 90, 1.5), (87, 80, 1.2),
    (78, 79, 2.5), (78, 93, 1.2), (79, 93, 1.0), (85, 78, 1.0), (88, 93, 1.0),
    (59, 61, 2.0), (59, 63, 1.5), (59, 64, 1.5), (59, 68, 1.2),
    (60, 61, 1.8), (60, 62, 1.5), (67, 59, 1.0),
    (39, 40, 1.2), (39, 48, 1.0), (44, 47, 1.0),
    (8, 90, 1.8), (6, 89, 1.2), (7, 89, 1.2), (8, 95, 1.0),
    (29, 33, 1.2),
]


class HybridRecommender:
    def __init__(self):
        self.products = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        self.id_to_idx = {p["id"]: i for i, p in enumerate(self.products)}
        self.n_items = len(self.products)

        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self):

        corpus = [
            f"{p['name']} {p['brand']} {p['category']} "
            f"{' '.join(p.get('tags', []))} {p.get('description', '')}"
            for p in self.products
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.content_sim = cosine_similarity(tfidf_matrix)

    ACCESSORY_CATEGORY_MAP = {
        "Consumer Electronics": ["Audio", "Office Supplies & Furniture"],
        "Mobiles & Tablets": ["Audio", "Fitness & Fashion Tech", "Office Supplies & Furniture"],
        "Audio": ["Mobiles & Tablets", "Consumer Electronics"],
        "Home Appliances": ["Kitchenware", "Home Automation"],
        "Kitchenware": ["Home Appliances"],
        "Personal Care": ["Fitness & Fashion Tech"],
        "Home Automation": ["Consumer Electronics", "Home Appliances"],
        "Fitness & Fashion Tech": ["Mobiles & Tablets", "Personal Care"],
        "Office Supplies & Furniture": ["Mobiles & Tablets", "Consumer Electronics"],
    }

    def _auto_fill_co_purchase_pairs(self, seed=7):
        
        covered = set()
        for a, b, _ in CO_PURCHASE_PAIRS:
            covered.add(a)
            covered.add(b)

        rng = np.random.default_rng(seed)
        cat_items_by_id = {}
        for p in self.products:
            cat_items_by_id.setdefault(p["category"], []).append(p["id"])

        auto_pairs = []
        for p in self.products:
            if p["id"] in covered:
                continue
            partner_cats = self.ACCESSORY_CATEGORY_MAP.get(p["category"], [])
            candidates = [
                pid for c in partner_cats for pid in cat_items_by_id.get(c, [])
            ]
            if not candidates:
                continue
            chosen = rng.choice(candidates, size=min(2, len(candidates)), replace=False)
            for pid in chosen:
                auto_pairs.append((p["id"], int(pid), 1.3))

        return auto_pairs

    def _build_collaborative_model(self, n_synthetic_users=400, n_components=35, seed=42):
        
        rng = np.random.default_rng(seed)
        categories = sorted({p["category"] for p in self.products})
        cat_items = {
            c: [i for i, p in enumerate(self.products) if p["category"] == c]
            for c in categories
        }

        affinity = {
            "Consumer Electronics": ["Audio", "Mobiles & Tablets"],
            "Mobiles & Tablets": ["Fitness & Fashion Tech", "Audio"],
            "Audio": ["Mobiles & Tablets", "Consumer Electronics"],
            "Home Appliances": ["Kitchenware", "Home Automation"],
            "Kitchenware": ["Home Appliances"],
            "Personal Care": ["Fitness & Fashion Tech"],
            "Home Automation": ["Consumer Electronics", "Home Appliances"],
            "Fitness & Fashion Tech": ["Mobiles & Tablets", "Personal Care"],
            "Office Supplies & Furniture": ["Mobiles & Tablets", "Consumer Electronics"],
        }

        all_pairs = CO_PURCHASE_PAIRS + self._auto_fill_co_purchase_pairs()
        pair_idx = [
            (self.id_to_idx[a], self.id_to_idx[b], w)
            for a, b, w in all_pairs
            if a in self.id_to_idx and b in self.id_to_idx
        ]

        R = np.zeros((n_synthetic_users, self.n_items))
        for u in range(n_synthetic_users):
            primary = rng.choice(categories)
            secondary = rng.choice(affinity.get(primary, categories))
            
            for i in cat_items[primary]:
                R[u, i] += rng.poisson(3.0)
            for i in cat_items[secondary]:
                R[u, i] += rng.poisson(1.2)
                
            noise_items = rng.choice(self.n_items, size=5, replace=False)
            for i in noise_items:
                R[u, i] += rng.poisson(0.6)

            for ia, ib, w in pair_idx:
                if R[u, ia] > 0:
                    R[u, ib] += rng.poisson(w)
                if R[u, ib] > 0:
                    R[u, ia] += rng.poisson(w * 0.6)

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
        
        for rank, ev in enumerate(reversed(events)):
            idx = self.id_to_idx.get(ev.get("product_id"))
            if idx is None:
                continue
            recency_decay = 0.5 ** (rank / RECENCY_HALF_LIFE)
            w = ACTION_WEIGHT.get(ev.get("action"), 1.0) * recency_decay
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
            idx = self.id_to_idx.get(ev.get("product_id"))
            if idx is None:
                continue
            recency_decay = 0.5 ** (rank / RECENCY_HALF_LIFE)
            w = ACTION_WEIGHT.get(ev.get("action"), 1.0) * recency_decay
            scores += w * self.content_sim[idx]
            total_w += w

        if total_w == 0:
            return None
            
        return scores / total_w

    def _dynamic_alpha(self, events):
        
        if not events:
            return 0.7
            
        last_action = events[-1].get("action", "view")
        depth_penalty = min(len(events), 6) * 0.05
        alpha = 0.75 - depth_penalty
        
        if last_action == "cart":
            alpha -= 0.20
            
        return max(0.25, min(0.80, alpha))

    def recommend(self, events, cart_ids=None, top_n=6):
        
        cart_ids = set(cart_ids or [])
        recent_ids = {ev["product_id"] for ev in events[-3:] if "product_id" in ev} if events else set()
        exclude = cart_ids | recent_ids

        content = self._content_score(events)
        cf_vec = self._fold_in_session(events)
        cf = None
        
        if cf_vec is not None:
            cf = self._item_factors_unit @ cf_vec

        if content is None and cf is None:
            # Cold-start fallback
            ranked = list(range(self.n_items))
            ranked.sort(key=lambda i: -self.products[i]["price"] % 7)
            reason = "popular"
            alpha_val = None
        else:
            alpha_val = self._dynamic_alpha(events)
            if content is None:
                blended = cf
                reason = "collaborative"
            elif cf is None:
                blended = content
                reason = "content"
            else:
                blended = alpha_val * content + (1 - alpha_val) * cf
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

        return results, alpha_val


engine = HybridRecommender()