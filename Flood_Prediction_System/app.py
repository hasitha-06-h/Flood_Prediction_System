"""
app.py

Main Flask application. Deliberately kept as a single module with plain
route functions rather than Blueprints - the site only has seven pages,
and splitting that across multiple Blueprint files would have added
indirection without buying anything at this scale.

Run locally:
    python train_model.py     (only needed once, or after changing the dataset)
    python app.py

Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime

from config import Config
import database
import utils
import prediction

app = Flask(__name__)
app.config.from_object(Config)


@app.before_request
def _ensure_database_ready():
    # init_db() is idempotent (CREATE TABLE IF NOT EXISTS) so calling it
    # on every request is wasteful long-term, but for a project this size
    # it guarantees the tables exist even on a very first cold request
    # after deployment, with zero extra setup steps for whoever runs it.
    database.init_db()


@app.route("/")
def home():
    metadata = utils.load_model_metadata()
    stats = database.count_predictions_by_risk()
    return render_template("home.html", metadata=metadata, stats=stats)


@app.route("/about")
def about():
    metadata = utils.load_model_metadata()
    return render_template("about.html", metadata=metadata)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", form_values=None, errors=None)

    try:
        parsed_form = utils.parse_and_validate_form(request.form)
    except utils.ValidationError as exc:
        return render_template(
            "predict.html",
            form_values=request.form,
            errors=str(exc),
        )

    model_reading = utils.to_model_reading(parsed_form)

    try:
        result = prediction.predict_flood(model_reading)
    except prediction.ModelNotReadyError as exc:
        return render_template(
            "predict.html",
            form_values=request.form,
            errors=str(exc),
        )

    metadata = utils.load_model_metadata()
    database.save_reading_and_prediction(
        reading=model_reading,
        model_used=metadata.get("model_name", "Unknown"),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
    )

    return render_template(
        "result.html",
        parsed_form=parsed_form,
        result=result,
        prediction_date=datetime.now().strftime("%d %b %Y, %I:%M %p"),
    )


@app.route("/alert")
def flood_alert():
    """Dedicated full-page emergency view, linked from the result page
    when a Severe/High risk reading is returned."""
    return render_template("chance.html")


@app.route("/safe")
def safe_page():
    """Dedicated full-page reassurance view for Low/Moderate readings."""
    return render_template("no_chance.html")


@app.route("/history")
def history():
    records = database.fetch_history(limit=100)
    stats = database.count_predictions_by_risk()
    return render_template("history.html", records=records, stats=stats)


@app.errorhandler(404)
def page_not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    database.init_db()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
