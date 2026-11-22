(() => {
  const search = document.querySelector("#similarity-search");
  const restart = document.querySelector("#similarity-restart");
  const results = document.querySelector("#similarity-results");
  const status = document.querySelector("#similarity-status");
  const iteration = document.querySelector("#similarity-iteration");
  const card = document.querySelector("#similarity-card");
  const feedback = document.querySelector("#similarity-feedback");
  const feedbackControls = document.querySelector("#similarity-feedback-controls");
  const selectionCount = document.querySelector("#similarity-selection-count");
  const fieldset = feedback.querySelector("fieldset");
  const submit = feedback.querySelector("button");
  const submitLabel = submit.querySelector("[data-feedback-label]");
  const distanceFormat = new Intl.NumberFormat("fr-FR", { maximumSignificantDigits: 7 });
  let sessionId = null;

  function updateFeedback() {
    const selected = results.querySelectorAll("input:checked").length;
    feedbackControls.hidden = !sessionId || !results.childElementCount;
    submit.disabled = selected === 0;
    selectionCount.textContent = selected === 1 ? "1 image sélectionnée" : `${selected} images sélectionnées`;
  }

  function renderResults(matches) {
    results.replaceChildren(...matches.map((match, index) => {
      const fragment = card.content.cloneNode(true);
      const image = fragment.querySelector("img");
      const link = fragment.querySelector("[data-full-image]");
      const missing = fragment.querySelector("[data-missing]");
      const checkbox = fragment.querySelector("input");
      const label = `Image GHIM n° ${match.image_id}`;
      const url = search.dataset.imageUrl.replace(/0$/, String(match.image_id));
      image.addEventListener("error", () => {
        image.classList.add("invisible");
        link.removeAttribute("href");
        missing.hidden = false;
      }, { once: true });
      image.alt = label;
      link.href = url;
      image.src = url;
      checkbox.value = String(match.image_id);
      checkbox.setAttribute("aria-label", `Marquer comme pertinente · ${label}`);
      fragment.querySelector("[data-rank]").textContent = `Rang ${index + 1}`;
      fragment.querySelector("[data-category]").textContent = match.category ?? "Sans catégorie";
      if (Number.isFinite(match.distance)) {
        const distance = fragment.querySelector("[data-distance]");
        distance.textContent = `Distance ${distanceFormat.format(match.distance)}`;
        distance.title = String(match.distance);
        distance.hidden = false;
      }
      return fragment;
    }));
  }

  async function requestMatches(url, payload, refining) {
    restart.disabled = true;
    fieldset.disabled = true;
    results.setAttribute("aria-busy", "true");
    status.dataset.state = "loading";
    if (refining) submitLabel.textContent = "Relance en cours…";
    status.textContent = refining ? "Retour de pertinence en cours…" : "Recherche en cours…";
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (response.redirected) {
        sessionId = null;
        status.dataset.state = "error";
        status.textContent = "Reconnectez-vous pour reprendre la recherche.";
        return;
      }
      const data = await response.json();
      if (!response.ok) {
        status.dataset.state = "error";
        if (!refining && response.status === 404) {
          status.textContent = "Cette image ne figure pas dans le lot indexé.";
        } else if (refining && response.status === 404) {
          sessionId = null;
          status.textContent = "La session a expiré. Relancez la recherche.";
        } else {
          if (response.status === 409) {
            sessionId = null;
          }
          status.textContent = data.error || "La recherche a échoué.";
        }
        return;
      }
      renderResults(data.results);
      sessionId = data.session_id;
      iteration.hidden = !Number.isInteger(data.iteration);
      iteration.textContent = iteration.hidden ? "" : `Itération ${data.iteration}`;
      const count = data.results.length;
      status.dataset.state = count ? "ready" : "empty";
      status.textContent = count ? `${count} résultat${count === 1 ? "" : "s"}` : "Aucune image similaire trouvée.";
    } catch {
      status.dataset.state = "error";
      status.textContent = "Le moteur de recherche est indisponible. Réessayez dans un instant.";
    } finally {
      submitLabel.textContent = "Relancer avec la sélection";
      results.removeAttribute("aria-busy");
      restart.disabled = false;
      fieldset.disabled = !sessionId;
      updateFeedback();
    }
  }

  function startSearch() {
    sessionId = null;
    iteration.hidden = true;
    results.replaceChildren();
    updateFeedback();
    return requestMatches(search.dataset.searchUrl, {
      query_image_id: Number(search.dataset.queryId),
      exclude_query: true,
    }, false);
  }

  feedback.addEventListener("change", updateFeedback);
  feedback.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!sessionId || fieldset.disabled || submit.disabled) {
      return;
    }
    const relevantIds = Array.from(results.querySelectorAll("input:checked"), (input) => Number(input.value));
    requestMatches(search.dataset.feedbackUrl, { session_id: sessionId, relevant_ids: relevantIds }, true);
  });
  restart.addEventListener("click", startSearch);
  startSearch();
})();
