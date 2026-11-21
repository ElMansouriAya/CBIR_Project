(() => {
  function icon(name) {
    const element = document.createElement("i");
    element.className = "bi bi-" + name;
    element.setAttribute("aria-hidden", "true");
    return element;
  }

  function searchableText(text) {
    return text.normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase("fr").trim();
  }

  for (const select of document.querySelectorAll("select[data-menu-select]")) {
    if (select.multiple || !select.options.length || select.disabled) continue;

    const label = select.labels[0];
    const caption = label?.textContent.trim() || select.getAttribute("aria-label");
    const root = document.createElement("div");
    root.className = "studio-select";

    const trigger = document.createElement("button");
    trigger.id = select.id + "-menu";
    trigger.type = "button";
    trigger.className = "btn btn-outline-secondary studio-select-button";
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-haspopup", "dialog");
    const value = document.createElement("span");
    value.className = "studio-select-value";
    trigger.append(icon(select.dataset.menuSelect), value, icon("chevron-down"));

    const panel = document.createElement("div");
    panel.id = select.id + "-panel";
    panel.className = "studio-menu";
    panel.hidden = true;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", caption || "Choisir une option");
    trigger.setAttribute("aria-controls", panel.id);

    let search;
    if (select.dataset.menuSearch) {
      const searchBox = document.createElement("div");
      searchBox.className = "studio-menu-search";
      search = document.createElement("input");
      search.type = "search";
      search.autocomplete = "off";
      search.placeholder = select.dataset.menuSearch;
      search.setAttribute("aria-label", search.placeholder);
      searchBox.append(icon("search"), search);
      panel.append(searchBox);
    }

    const menu = document.createElement("div");
    menu.className = "studio-options";
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", caption || "Options");
    const choices = Array.from(select.options, (option, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "studio-option";
      button.disabled = option.disabled;
      button.tabIndex = -1;
      button.setAttribute("role", "option");
      const text = document.createElement("span");
      text.textContent = option.textContent.trim();
      button.append(text, icon("check-lg"));
      button.addEventListener("click", () => {
        select.selectedIndex = index;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        close(true);
      });
      menu.append(button);
      return button;
    });
    const empty = document.createElement("p");
    empty.className = "studio-menu-empty";
    empty.setAttribute("role", "status");
    empty.textContent = "Aucune catégorie trouvée.";
    empty.hidden = true;
    panel.append(menu, empty);

    function sync() {
      const text = select.options[select.selectedIndex]?.textContent.trim() || "";
      value.textContent = text;
      trigger.setAttribute("aria-label", caption ? caption + " : " + text : text);
      choices.forEach((option, index) => {
        option.setAttribute("aria-selected", String(index === select.selectedIndex));
      });
    }

    function placePanel() {
      if (panel.hidden) return;
      const anchor = trigger.getBoundingClientRect();
      const viewport = window.visualViewport;
      const pageBottom = (viewport?.offsetTop ?? 0) + (viewport?.height ?? window.innerHeight);
      const topbar = document.querySelector(".topbar")?.getBoundingClientRect().bottom || 0;
      const below = pageBottom - anchor.bottom - 12;
      const above = anchor.top - Math.max(topbar, viewport?.offsetTop ?? 0) - 12;
      const desired = menu.scrollHeight + (search?.parentElement.offsetHeight ?? 0) + empty.offsetHeight + 12;
      const opensAbove = below < Math.min(desired, 304) && above > below;
      const room = opensAbove ? above : below;
      const maxLeft = document.documentElement.clientWidth - panel.offsetWidth - 12;
      const left = Math.max(12, Math.min(anchor.left, maxLeft));
      panel.classList.toggle("opens-above", opensAbove);
      panel.style.setProperty("--studio-menu-room", Math.max(0, room) + "px");
      panel.style.setProperty("--studio-menu-left", left - anchor.left + "px");
    }

    function filterOptions() {
      const term = searchableText(search?.value || "");
      for (const choice of choices) choice.hidden = !searchableText(choice.textContent).includes(term);
      empty.hidden = choices.some(choice => !choice.hidden);
      menu.scrollTop = 0;
      placePanel();
    }

    function revealChoice(choice) {
      if (!choice) return;
      const box = choice.getBoundingClientRect();
      const scroller = menu.getBoundingClientRect();
      if (box.top < scroller.top) menu.scrollTop -= scroller.top - box.top;
      if (box.bottom > scroller.bottom) menu.scrollTop += box.bottom - scroller.bottom;
    }

    function focusChoice(choice) {
      choice?.focus({ preventScroll: true });
      revealChoice(choice);
    }

    function open(touch = false, last = false) {
      panel.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      if (search) search.value = "";
      filterOptions();
      if (search) {
        revealChoice(choices[select.selectedIndex]);
        if (!touch) search.focus({ preventScroll: true });
      } else {
        const enabled = choices.filter(choice => !choice.disabled);
        focusChoice(last ? enabled.at(-1) : enabled.find(choice => choice.getAttribute("aria-selected") === "true") || enabled[0]);
      }
    }

    function close(restoreFocus = false) {
      if (panel.hidden) return;
      panel.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      if (restoreFocus) trigger.focus({ preventScroll: true });
    }

    trigger.addEventListener("click", event => {
      if (panel.hidden) open(event.pointerType === "touch");
      else close(true);
    });
    search?.addEventListener("input", filterOptions);
    root.addEventListener("keydown", event => {
      const { key, target } = event;
      if (key === "Escape" && !panel.hidden) {
        event.preventDefault();
        event.stopPropagation();
        close(true);
      } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(key)) {
        if ((target === search || target === trigger) && (key === "Home" || key === "End")) return;
        event.preventDefault();
        if (panel.hidden) {
          open(false, key === "ArrowUp");
          return;
        }
        const visible = choices.filter(choice => !choice.hidden && !choice.disabled);
        const index = visible.indexOf(target);
        if (key === "ArrowUp" && index === 0 && search) search.focus({ preventScroll: true });
        else if (key === "Home" || (key === "ArrowDown" && index < 0)) focusChoice(visible[0]);
        else if (key === "End" || (key === "ArrowUp" && index < 0)) focusChoice(visible.at(-1));
        else focusChoice(visible[index + (key === "ArrowDown" ? 1 : -1)]);
      } else if (key === "Enter" && target === search) {
        event.preventDefault();
        choices.find(choice => !choice.hidden && !choice.disabled)?.click();
      }
    });
    root.addEventListener("focusout", event => {
      if (event.relatedTarget && !root.contains(event.relatedTarget)) close();
    });
    document.addEventListener("pointerdown", event => {
      if (!root.contains(event.target)) close();
    });
    window.addEventListener("resize", placePanel);
    window.addEventListener("scroll", placePanel, { passive: true });
    window.visualViewport?.addEventListener("resize", placePanel);

    root.append(trigger, panel);
    select.after(root);
    select.hidden = true;
    if (label) label.htmlFor = trigger.id;
    select.addEventListener("change", sync);
    select.form?.addEventListener("reset", () => requestAnimationFrame(sync));
    sync();
  }

  function showPress(event) {
    const action = event.target.closest(".btn-primary, .photo-primary");
    if (!action || action.matches(":disabled, [aria-disabled='true']")) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const bounds = action.getBoundingClientRect();
    const halo = document.createElement("span");
    halo.className = "action-halo";
    halo.setAttribute("aria-hidden", "true");
    const centered = event.type === "click";
    halo.style.left = (centered ? bounds.width / 2 : event.clientX - bounds.left) + "px";
    halo.style.top = (centered ? bounds.height / 2 : event.clientY - bounds.top) + "px";
    action.append(halo);
    halo.addEventListener("animationend", () => halo.remove(), { once: true });
  }

  document.addEventListener("pointerdown", event => {
    if (event.button === 0) showPress(event);
  });
  document.addEventListener("click", event => {
    if (event.detail === 0) showPress(event);
  });
})();
