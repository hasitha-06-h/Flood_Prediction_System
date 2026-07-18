/*
  validation.js
  Client-side mirror of utils.parse_and_validate_form() on the server.
  This is a UX nicety only - the server re-validates everything, so an
  attacker bypassing this script gains nothing.
*/

const FIELD_RULES = [
  { name: "annual_rainfall", label: "Annual Rainfall", min: 0, max: 10000 },
  { name: "cloud_cover", label: "Cloud Visibility", min: 0, max: 100 },
  { name: "temperature", label: "Temperature", min: -10, max: 60 },
  { name: "humidity", label: "Humidity", min: 0, max: 100 },
  { name: "seasonal_rainfall", label: "Seasonal Rainfall", min: 0, max: 5000 },
];

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("flood-predict-form");
  if (!form) return;

  form.addEventListener("submit", function (event) {
    let firstInvalidField = null;

    FIELD_RULES.forEach((rule) => {
      const input = form.querySelector(`[name="${rule.name}"]`);
      if (!input) return;

      const value = parseFloat(input.value);
      const isValid =
        input.value.trim() !== "" &&
        !Number.isNaN(value) &&
        value >= rule.min &&
        value <= rule.max;

      input.classList.toggle("is-invalid", !isValid);
      if (!isValid && !firstInvalidField) {
        firstInvalidField = input;
      }
    });

    if (firstInvalidField) {
      event.preventDefault();
      firstInvalidField.focus();
    } else {
      toggleSpinner(true);
    }
  });

  form.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => input.classList.remove("is-invalid"));
  });
});

function toggleSpinner(show) {
  const spinner = document.getElementById("predict-spinner");
  const button = document.getElementById("predict-submit-btn");
  if (!spinner || !button) return;

  spinner.classList.toggle("d-none", !show);
  button.disabled = show;
  button.querySelector(".btn-label").textContent = show ? "Analysing weather data..." : "Predict Flood Risk";
}
