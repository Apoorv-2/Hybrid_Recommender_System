(() => {
  const HOVER_DWELL_MS = 350; // avoid firing on accidental mouse-overs

  const recList = document.getElementById("rec-list");
  const signalSource = document.getElementById("signal-source");
  const cartCountEl = document.getElementById("cart-count");
  const signalDot = document.querySelector(".signal-dot");

  const reasonCopy = {
    content: "similar item",
    collaborative: "customers also bought",
    hybrid: "similar + also bought",
    popular: "popular pick",
  };

  function renderRecs(data) {
    if (!recList) return;
    recList.innerHTML = "";
    data.recommendations.forEach((p) => {
      const li = document.createElement("li");
      li.className = "rec-item";
      li.dataset.id = p.id;
      li.innerHTML = `
        <span class="cable-glyph" aria-hidden="true"></span>
        <div class="rec-body">
          <a href="/product/${p.id}">${p.name}</a>
          <span class="rec-meta">${p.brand} · Rs.${Math.round(p.price)}</span>
        </div>
        <span class="rec-tag rec-tag-${p.reason}">${reasonCopy[p.reason] || p.reason}</span>
      `;
      recList.appendChild(li);
    });

    if (signalSource && data.focus) {
      const verb = data.action === "cart" ? "added to cart" : "looking at";
      signalSource.innerHTML = `Signal patched from <strong>${data.focus.name}</strong> (${verb}).`;
    }

    if (cartCountEl) cartCountEl.textContent = data.cart_count;

    if (signalDot) {
      signalDot.classList.add("hot");
      setTimeout(() => signalDot.classList.remove("hot"), 900);
    }
  }

  async function track(productId, action) {
    try {
      const res = await fetch("/api/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: productId, action }),
      });
      if (!res.ok) return;
      const data = await res.json();
      renderRecs(data);
    } catch (err) {
      // Fail silently -- recommendations are an enhancement, not critical path
      console.warn("track() failed", err);
    }
  }

  // --- Hover-with-dwell = "view" signal ---
  document.querySelectorAll(".card[data-id]").forEach((card) => {
    let timer = null;
    card.addEventListener("mouseenter", () => {
      timer = setTimeout(() => track(card.dataset.id, "view"), HOVER_DWELL_MS);
    });
    card.addEventListener("mouseleave", () => {
      if (timer) clearTimeout(timer);
    });
  });

  // --- Add to cart = strongest signal ---
  document.querySelectorAll(".btn-add-cart[data-cart-id]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.cartId;
      btn.disabled = true;
      const original = btn.textContent;
      btn.textContent = "added ✓";
      btn.classList.add("just-added");
      await track(id, "cart");
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove("just-added");
        btn.disabled = false;
      }, 1200);
    });
  });

  // --- Reset session ---
  const resetBtn = document.getElementById("reset-session");
  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      await fetch("/api/reset", { method: "POST" });
      window.location.reload();
    });
  }
})();
