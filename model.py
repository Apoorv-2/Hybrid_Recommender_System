from asyncio import events
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

DATA_PATH = Path(__file__).parent/ "data" / "product.json"

ACTION_WEIGHT = {
    "view" : 1.0,
    "click" : 2.0,
    "cart" : 3.5
}

RECENCY_HALF_LIFE = 4

class HybridRecommender:
    def __init__(self , data_path = DATA_PATH):
        self.products = self.load_products(data_path)
        self.id_to_idx = {product["id"] : idx for idx , product in enumerate(self.products)}
        self.n_items = len(self.products)

        self.build_content_model()
        self.build_collaborative_model()

    def build_content_model(self):
        data = [
            f"{p['name']} {p['brand']} {p['category']}"
            f"{" ".join(p['tags'])} {p['description']}"
            for p in self.products
        ]
        self.vectorizer = TfidfVectorizer(stop_words = "english")
        tfidf_matrix = self.vectorizer.fit_transform(data)
        self.content_similarity = cosine_similarity(tfidf_matrix)

    def build_collaborative_model(self , n_virtual_users = 400 , n_components = 10 , seed = 42):
        rng = np.random.default_rng(seed)
        categories = sorted({p["category"] for p in self.products})
        cart_items = {c : [i for i , p in enumerate(self.products) if p["category"] == c] for c in categories}

        affinity = {
            "Synthesizers": ["Drum Machines", "Cables & Accessories"],
            "Drum Machines": ["Synthesizers", "Effects Pedals"],
            "Effects Pedals": ["Drum Machines", "Cables & Accessories"],
            "Eurorack Modules": ["Cables & Accessories", "Synthesizers"],
            "Cables & Accessories": ["Eurorack Modules"],
        }

        R = np.zeros((n_virtual_users , self.n_items))
        for u in range(n_virtual_users):
            primary = rng.choice(categories)
            secondary = rng.choice(affinity.get(primary , categories))
            for i in cart_items[primary]:
                R[u , i] += rng.poisson(3.0)
            for i in cart_items[secondary]:
                R[u , i] += rng.poisson(1.2)
            noise = rng.choice(self.n_items , size =  3 , replace = False)
            for i in noise:
                R[u , i] += rng.poisson(0.4)
            
        
        self.svd = TruncatedSVD(n_components = n_virtual_users , random_state = seed)
        self.user_factor = self.svd.fit_transform(R)
        self.item_factor = self.svd.components_.T 

        normalization = np.linalg.norm(self.item_factor , axis = 1 , keepdims=True)
        normalization[normalization == 0] = 1e-9
        self.item_factor_unit = self.item_factor / normalization

        def online_session(self , events):
            if not events:
                return None
            vec = np.zeros(self.item_factor.shape[1])
            total_w = 0.0
            n = len(events)
            for rank , ev in enumerate(reversed(events)):
                idx = self.id_to_idx.get(ev["product_id"])
                if idx is None:
                    continue
                recency_decay = 0.5 ** (rank/RECENCY_HALF_LIFE)
                w = ACTION_WEIGHT.get(ev["action"] , 1.0) * recency_decay
                vec += w * self.item_factor[idx]
                total_w += w
            if total_w == 0:
                return None
            vec /= total_w  #centroid averaging
            normalization = np.linalg.norm(vec)
            if normalization == 0:
                return None
            return vec / normalization # euclidean (L2) normalization
        
        def content_score(self , events):
            if not events:
                return None
            scores = np.zeros(self.n_items)
            total_w = 0.0
            for rank , ev in enumerate(reversed(events)):
                idx = self.id_to_idx.get(ev["product_id"])
                if idx is None:
                    continue
                recency_decay = 0.5 ** (rank/RECENCY_HALF_LIFE)
                w = ACTION_WEIGHT.get(ev["action"] , 1.0) * recency_decay
                scores += w * self.content_similarity[idx]
                total_w += w
            if total_w == 0:
                return None
            return scores / total_w
        
        def alpha(self , events):
            if not events:
                return 0.7
            last_action = events[-1]["action"]
            depth_penalty = min(len(events) , 5) * 0.05
            alpha = 0.75 - depth_penalty
            if last_action == "cart":
                alpha -= 0.2
            return max(0.25 , min(0.8 , alpha))
        
        def recommend(self , events , cart_ids = None , top_n = 6):
            cart_ids = set(cart_ids or [])
            recent_ids = {ev["product_id"] for ev in events[-3:]} if events else set()
            exclude = cart_ids | recent_ids #union
            content = self.content_score(events)
            cf_vec = self.online_session(events)
            cf = None
            if cf_vec is not None:
                cf = self.item_factor_unit @ cf_vec
            
            if content is None and cf is None:
                ranked = list(range(self.n_items))
                ranked.sort(key = lambda i : -self.products[i]["price"] % 7)
                reason = "popular"
            
            else:
                alpha = self.alpha(events)
                if content is None:
                    blended = cf
                    reason = "collaborative"
                elif cf is None:
                    blended = content
                    reason = "content"
                else:
                    blended = alpha * content + (1-alpha) * cf
                    reason = "hybrid"
                ranked = list(np.argsort(-blended))
            
            result = []
            for idx in ranked :
                pid = self.products[idx][id]
                if pid in exclude:
                    continue
                item = dict(self.products[idx])
                item["reason"] = reason
                result.append(item)
                if len(result) >= top_n:
                    break
            return result , (self.alpha(events) if events else None)
        

engine = HybridRecommender()
