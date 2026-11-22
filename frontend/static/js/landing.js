const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function setupHeader() {
  const header = document.querySelector(".landing-header");
  const toggle = header.querySelector(".landing-menu-toggle");
  const panel = header.querySelector(".landing-menu");
  const mobile = window.matchMedia("(width < 860px)");
  const featureLink = header.querySelector(".landing-nav a");
  const features = document.getElementById(featureLink.hash.slice(1));

  function setMenu(open) {
    panel.toggleAttribute("data-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Fermer le menu" : "Ouvrir le menu");
  }

  header.dataset.menuReady = "";
  toggle.hidden = false;
  toggle.addEventListener("click", () => setMenu(!panel.hasAttribute("data-open")));
  mobile.addEventListener("change", () => setMenu(false));
  panel.addEventListener("click", event => {
    const link = event.target.closest("a");
    if (!link || !mobile.matches || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    setMenu(false);
    if (link.hash && link.pathname === location.pathname) {
      const section = document.getElementById(link.hash.slice(1));
      section?.focus({ preventScroll: true });
    }
  });
  document.addEventListener("keydown", event => {
    if (event.key !== "Escape" || !panel.hasAttribute("data-open")) return;
    setMenu(false);
    toggle.focus();
  });

  function updateHeader() {
    header.classList.toggle("is-scrolled", window.scrollY > 20);
  }

  const observer = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) featureLink.setAttribute("aria-current", "location");
    else featureLink.removeAttribute("aria-current");
  }, { rootMargin: "-88px 0px 0px 0px" });
  observer.observe(features);

  let frame;
  window.addEventListener("scroll", () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = undefined;
      updateHeader();
    });
  }, { passive: true });
  updateHeader();
}

function setupHistogram() {
  const card = document.querySelector(".landing-descriptor");
  const chart = card.querySelector(".landing-bars");
  const hues = [
    { value: 9, height: 30, color: "#C9B3FA" },
    { value: 17, height: 55, color: "#A98AF0" },
    { value: 31, height: 100, color: "#7B4DDB" },
    { value: 20, height: 65, color: "#E58DBA" },
    { value: 13, height: 40, color: "#F2B8D6" },
    { value: 6, height: 20, color: "#C9B3FA" },
  ];
  const bars = hues.map(({ value, height, color }, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "landing-bar";
    button.setAttribute("aria-label", "Teinte " + (index + 1) + ", " + value + " %");
    button.setAttribute("aria-pressed", "false");
    button.style.setProperty("--bar-height", height + "%");
    button.style.setProperty("--bar-color", color);
    const shape = document.createElement("span");
    shape.className = "landing-bar-shape";
    shape.setAttribute("aria-hidden", "true");
    button.append(shape);
    chart.append(button);
    return { button, shape, value };
  });

  const tooltip = document.createElement("span");
  tooltip.className = "landing-bar-value";
  tooltip.setAttribute("aria-hidden", "true");
  let selected, hideTimer;
  let animations = [];

  function stopAnimation() {
    for (const animation of animations) animation.cancel();
    animations = [];
  }

  function replay() {
    stopAnimation();
    if (reducedMotion.matches) return;
    animations = bars.map(({ shape }, index) => shape.animate(
      [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }],
      { duration: 600, delay: index * 70, easing: "cubic-bezier(.34,1.56,.64,1)", fill: "backwards" },
    ));
  }

  card.addEventListener("click", event => {
    const choice = bars.find(({ button }) => button === event.target.closest(".landing-bar"));
    clearTimeout(hideTimer);
    tooltip.remove();
    if (choice) {
      selected?.button.setAttribute("aria-pressed", "false");
      selected = choice;
      choice.button.setAttribute("aria-pressed", "true");
      tooltip.textContent = choice.value + " %";
      choice.button.append(tooltip);
      hideTimer = setTimeout(() => tooltip.remove(), 2600);
    }
    replay();
  });

  card.hidden = false;
  const observer = new IntersectionObserver(([entry]) => {
    if (entry.intersectionRatio < .4) return;
    observer.disconnect();
    replay();
  }, { threshold: .4 });
  observer.observe(card);
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) stopAnimation();
  });
}

function showRipple(event) {
  const button = event.target.closest(".landing-button-primary, .landing-button-white");
  if (!button || reducedMotion.matches || event.button !== 0) return;
  const rect = button.getBoundingClientRect();
  const fromKeyboard = event.type === "click";
  const ripple = document.createElement("span");
  ripple.className = "landing-ripple";
  ripple.setAttribute("aria-hidden", "true");
  ripple.style.setProperty("--ripple-x", (fromKeyboard ? rect.width / 2 : event.clientX - rect.left) + "px");
  ripple.style.setProperty("--ripple-y", (fromKeyboard ? rect.height / 2 : event.clientY - rect.top) + "px");
  ripple.style.setProperty("--ripple-size", Math.hypot(rect.width, rect.height) * 2 + "px");
  button.append(ripple);
  setTimeout(() => ripple.remove(), 600);
}

setupHeader();
setupHistogram();
document.addEventListener("pointerdown", showRipple);
document.addEventListener("click", event => {
  if (event.detail === 0) showRipple(event);
});
