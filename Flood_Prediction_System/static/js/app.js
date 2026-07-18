/*
  app.js
  Small, dependency-free interactions shared across every page:
  - fade-in-on-scroll for sections tagged .fade-in-section
  - animated count-up for the hero statistics
  - active nav-link highlighting based on current path
  No frameworks - the site is server-rendered Jinja, so this stays vanilla.
*/

document.addEventListener("DOMContentLoaded", function () {
  highlightActiveNavLink();
  setupScrollReveal();
  setupCounters();
});

function highlightActiveNavLink() {
  const currentPath = window.location.pathname;
  document.querySelectorAll(".navbar-flood .nav-link").forEach((link) => {
    const linkPath = new URL(link.href).pathname;
    if (linkPath === currentPath) {
      link.classList.add("active");
    }
  });
}

function setupScrollReveal() {
  const sections = document.querySelectorAll(".fade-in-section");
  if (!("IntersectionObserver" in window) || sections.length === 0) {
    sections.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  sections.forEach((section) => observer.observe(section));
}

function setupCounters() {
  const counters = document.querySelectorAll("[data-count-to]");
  counters.forEach((el) => {
    const target = parseFloat(el.dataset.countTo);
    const suffix = el.dataset.countSuffix || "";
    const durationMs = 1200;
    const startTime = performance.now();

    function step(now) {
      const progress = Math.min((now - startTime) / durationMs, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
      const current = (target * eased).toFixed(target % 1 !== 0 ? 1 : 0);
      el.textContent = current + suffix;
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.classList.add("counted");
      }
    }
    requestAnimationFrame(step);
  });
}
