for (const toggle of document.querySelectorAll(".password-toggle")) {
  const input = document.getElementById(toggle.getAttribute("aria-controls"));
  toggle.hidden = false;
  toggle.addEventListener("click", () => {
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    toggle.toggleAttribute("data-revealed", show);
    toggle.setAttribute("aria-label", show ? "Masquer le mot de passe" : "Afficher le mot de passe");
  });
}

for (const close of document.querySelectorAll(".flash-close")) {
  close.hidden = false;
  close.addEventListener("click", () => {
    const flash = close.closest(".login-flash");
    const next = flash.nextElementSibling?.querySelector(".flash-close")
      || flash.previousElementSibling?.querySelector(".flash-close")
      || document.querySelector(".login-form input");
    flash.remove();
    next.focus();
  });
}
