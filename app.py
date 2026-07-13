import os
import time
from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv 

from model import engine

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

MAX_EVENTS = 25


def get_events():
    return session.get("events", [])


def push_event(product_id, action):
    events = get_events()
    events.append({"product_id": product_id, "action": action, "t": time.time()})
    session["events"] = events[-MAX_EVENTS:]
    session.modified = True


def get_cart():
    return session.get("cart", [])


@app.route("/")
def index():
    recs, alpha = engine.recommend(get_events(), get_cart(), top_n=6)
    return render_template(
        "index.html",
        products=engine.products,
        recs=recs,
        cart_count=len(get_cart()),
        alpha=alpha,
    )


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = next((p for p in engine.products if p["id"] == product_id), None)
    if product is None:
        return "Not found", 404
    push_event(product_id, "click")
    recs, alpha = engine.recommend(get_events(), get_cart(), top_n=4)
    return render_template(
        "product.html",
        product=product,
        recs=recs,
        cart_count=len(get_cart()),
        alpha=alpha,
    )


@app.route("/cart")
def cart():
    cart_ids = get_cart()
    items = [p for p in engine.products if p["id"] in cart_ids]
    return render_template("cart.html", items=items, cart_count=len(cart_ids))


# ---------------------------------------------------------------------
# Real-time API: the frontend calls this on hover/click/add-to-cart and
# swaps the "Recommended for You" shelf with the response, no reload.
# ---------------------------------------------------------------------
@app.route("/api/track", methods=["POST"])
def api_track():
    payload = request.get_json(force=True)
    product_id = int(payload.get("product_id"))
    action = payload.get("action", "view")

    if action not in ("view", "click", "cart"):
        action = "view"

    push_event(product_id, action)

    if action == "cart":
        cart_ids = get_cart()
        if product_id not in cart_ids:
            cart_ids.append(product_id)
            session["cart"] = cart_ids
            session.modified = True

    recs, alpha = engine.recommend(get_events(), get_cart(), top_n=6)
    focus = next((p for p in engine.products if p["id"] == product_id), None)

    return jsonify(
        {
            "focus": {"id": focus["id"], "name": focus["name"]} if focus else None,
            "action": action,
            "alpha": round(alpha, 2) if alpha is not None else None,
            "cart_count": len(get_cart()),
            "recommendations": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "brand": p["brand"],
                    "category": p["category"],
                    "price": p["price"],
                    "reason": p["_reason"],
                }
                for p in recs
            ],
        }
    )


@app.route("/api/reset", methods=["POST"])
def api_reset():
    session.pop("events", None)
    session.pop("cart", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
