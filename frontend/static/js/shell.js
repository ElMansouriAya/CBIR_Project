(() => {
  const shell = document.querySelector(".app-shell");
  const navToggle = document.querySelector(".site-nav-toggle");
  const account = document.querySelector(".site-account");
  shell.setAttribute("data-nav-ready", "");
  navToggle.hidden = false;
  function setNavigation(open) {
    shell.toggleAttribute("data-nav-open", open);
    navToggle.setAttribute("aria-expanded", open);
    navToggle.setAttribute("aria-label", open ? "Fermer la navigation" : "Ouvrir la navigation");
    if (!open) account.open = false;
  }
  navToggle.addEventListener("click", () => setNavigation(!shell.hasAttribute("data-nav-open")));
  document.addEventListener("click", event => {
    if (!account.contains(event.target)) account.open = false;
  });
  document.addEventListener("keydown", event => {
    if (event.defaultPrevented || event.key !== "Escape") return;
    if (account.open) {
      account.open = false;
      account.querySelector("summary").focus();
    } else if (shell.hasAttribute("data-nav-open") && navToggle.getClientRects().length) {
      setNavigation(false);
      navToggle.focus();
    }
  });
})();
