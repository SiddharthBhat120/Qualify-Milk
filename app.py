"""Qualify Milk: predict milk quality from 7 lab readings."""
import json
import os
import secrets
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import bcrypt
import joblib
import pandas as pd
from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ml" / "milk_model.joblib"
METRICS_PATH = ROOT / "ml" / "metrics.json"
DATA_PATH = ROOT / "data" / "milknew.csv"
(ROOT / "instance").mkdir(exist_ok=True)


def _secret_key():
    """Use SECRET_KEY from the environment, or create one and keep it in instance/."""
    if os.getenv("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    f = ROOT / "instance" / ".secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


app = Flask(__name__)
app.config.update(
    SECRET_KEY=_secret_key(),
    SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'instance' / 'qualify.db'}"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
db = SQLAlchemy(app)

# ── Database ──────────────────────────────────────────────────────────


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(128), unique=True, nullable=False)
    password = db.Column(db.LargeBinary(60), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    predictions = db.relationship("Prediction", backref="user", cascade="all, delete-orphan", lazy="dynamic")


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    label = db.Column(db.String(32))  # optional sample name, e.g. "Morning batch"
    inputs = db.Column(db.Text, nullable=False)  # JSON
    grade = db.Column(db.String(8), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    @property
    def values(self):
        return json.loads(self.inputs)


with app.app_context():
    db.create_all()

# ── Model ─────────────────────────────────────────────────────────────

FIELDS = {
    # name: (min, max, step, kind)
    "pH": (3.0, 9.5, 0.1, "number"),
    "Temperature": (34, 90, 1, "number"),
    "Taste": (0, 1, 1, "binary"),
    "Odor": (0, 1, 1, "binary"),
    "Fat": (0, 1, 1, "binary"),
    "Turbidity": (0, 1, 1, "binary"),
    "Colour": (240, 255, 1, "number"),
}
GRADES = ["low", "medium", "high"]


def _load_model():
    """Load the trained model. If it is missing or was saved by another scikit-learn
    version, quickly retrain the chosen model so the app always starts."""
    try:
        bundle = joblib.load(MODEL_PATH)
        bundle["model"].predict(pd.DataFrame([[6.6, 35, 1, 0, 1, 0, 254]], columns=bundle["features"]))
        return bundle
    except Exception as exc:  # noqa: BLE001
        print(f"[model] Could not load saved model ({exc.__class__.__name__}); retraining…")
        from ml.train import FEATURES, CLASSES, candidates, load
        _, X, y, _ = load()
        name = "Stacked Ensemble"
        if METRICS_PATH.exists():
            name = json.loads(METRICS_PATH.read_text()).get("best_model", name)
        model = candidates()[name].fit(X, y)
        bundle = {"model": model, "features": FEATURES, "classes": CLASSES, "name": name}
        joblib.dump(bundle, MODEL_PATH)
        return bundle


BUNDLE = _load_model()


def load_metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return None


def parse_inputs(src):
    """Validate the 7 readings. Returns (values, errors)."""
    values, errors = {}, {}
    for name, (lo, hi, _step, kind) in FIELDS.items():
        raw = src.get(name)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a number."
            continue
        if kind == "binary" and v not in (0, 1):
            errors[name] = "Must be 0 or 1."
        elif not lo <= v <= hi:
            errors[name] = f"Must be between {lo:g} and {hi:g}."
        values[name] = int(v) if kind == "binary" or name in ("Temperature", "Colour") else round(v, 2)
    return values, errors


def insights(v):
    """Plain-language notes about readings that commonly signal a problem."""
    notes = []
    if v["pH"] < 6.4:
        notes.append(("bad", "pH", f"pH {v['pH']} is acidic. Fresh milk sits around 6.4–6.8; lower usually means souring."))
    elif v["pH"] > 6.8:
        notes.append(("bad", "pH", f"pH {v['pH']} is alkaline. Above 6.8 can point to mastitis milk or added neutralisers."))
    else:
        notes.append(("good", "pH", f"pH {v['pH']} is in the normal 6.4–6.8 range for fresh milk."))
    if v["Temperature"] > 45:
        notes.append(("warn", "Temperature", f"{v['Temperature']}°C is hot. Most high-grade samples were recorded at 34–45°C."))
    else:
        notes.append(("good", "Temperature", f"{v['Temperature']}°C is in the range most high-grade samples share."))
    notes.append(("good", "Taste", "Taste is good.") if v["Taste"] else ("bad", "Taste", "Taste is off."))
    notes.append(("good", "Odor", "Smell is normal.") if v["Odor"] else ("bad", "Odor", "Smell is off."))
    notes.append(("good", "Fat", "Fat content is optimal.") if v["Fat"] else ("warn", "Fat", "Fat content is below optimal."))
    notes.append(("warn", "Turbidity", "Milk is cloudy (high turbidity).") if v["Turbidity"] else ("good", "Turbidity", "Milk is clear (low turbidity)."))
    if v["Colour"] < 250:
        notes.append(("warn", "Colour", f"Colour {v['Colour']} is off-white. Fresh milk scores 250–255."))
    else:
        notes.append(("good", "Colour", f"Colour {v['Colour']} is a clean white."))
    return [{"level": a, "field": b, "text": c} for a, b, c in notes]


def predict(values):
    X = pd.DataFrame([[values[f] for f in BUNDLE["features"]]], columns=BUNDLE["features"])
    proba = BUNDLE["model"].predict_proba(X)[0]
    idx = int(proba.argmax())
    return {
        "grade": GRADES[idx],
        "confidence": round(float(proba[idx]) * 100, 1),
        "probabilities": {g: round(float(p) * 100, 1) for g, p in zip(GRADES, proba)},
        "insights": insights(values),
        "model": BUNDLE["name"],
    }


def sample_presets():
    """One real sample of each grade from the dataset, for the 'try a sample' buttons."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Temprature": "Temperature"})
    picks = {"high": 0, "medium": 4, "low": 2}
    out = {}
    for grade in GRADES[::-1]:
        rows = df[df["Grade"].str.strip().str.lower() == grade]
        row = rows.iloc[min(picks[grade], len(rows) - 1)]
        out[grade] = {f: (float(row[f]) if f == "pH" else int(row[f])) for f in FIELDS}
    return out


PRESETS = sample_presets()

# ── Auth helpers ─────────────────────────────────────────────────────


def current_user():
    uid = session.get("uid")
    return db.session.get(User, uid) if uid else None


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            if request.path.startswith("/api/"):
                return jsonify(error="Log in to run predictions."), 401
            flash("Log in to use the analyser.", "info")
            return redirect(url_for("auth", next=request.path))
        return view(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_globals():
    return {"user": current_user(), "year": datetime.now().year}


def _safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else url_for("analyse")

# ── Pages ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html", metrics=load_metrics())


@app.route("/analyse")
@login_required
def analyse():
    return render_template("analyse.html", fields=FIELDS, presets=PRESETS, model_name=BUNDLE["name"])


@app.route("/models")
def models():
    m = load_metrics()
    if not m:
        return render_template("models.html", metrics=None)
    return render_template("models.html", metrics=m, best=next(r for r in m["models"] if r["name"] == m["best_model"]))


@app.route("/about")
def about():
    return render_template("about.html", metrics=load_metrics())


@app.route("/history")
@login_required
def history():
    preds = current_user().predictions.order_by(Prediction.created_at.desc()).limit(200).all()
    counts = {g: sum(p.grade == g for p in preds) for g in GRADES}
    return render_template("history.html", preds=preds, counts=counts)


@app.post("/history/<int:pid>/delete")
@login_required
def delete_prediction(pid):
    p = Prediction.query.filter_by(id=pid, user_id=session["uid"]).first_or_404()
    db.session.delete(p)
    db.session.commit()
    flash("Sample removed from history.", "success")
    return redirect(url_for("history"))


@app.post("/history/clear")
@login_required
def clear_history():
    Prediction.query.filter_by(user_id=session["uid"]).delete()
    db.session.commit()
    flash("History cleared.", "success")
    return redirect(url_for("history"))

# ── API ───────────────────────────────────────────────────────────────


@app.post("/api/predict")
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    values, errors = parse_inputs(data)
    if errors:
        return jsonify(error="Check the highlighted readings.", fields=errors), 400
    result = predict(values)
    if data.get("save"):
        label = str(data.get("label") or "").strip()[:32] or None
        p = Prediction(user_id=session["uid"], label=label, inputs=json.dumps(values),
                       grade=result["grade"], confidence=result["confidence"])
        db.session.add(p)
        db.session.commit()
        result["saved_id"] = p.id
    return jsonify(result)

# ── Auth ──────────────────────────────────────────────────────────────


@app.route("/auth", methods=["GET"])
def auth():
    if current_user():
        return redirect(url_for("analyse"))
    return render_template("auth.html", mode=request.args.get("mode", "login"), next=request.args.get("next", ""))


@app.post("/login")
def login():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    nxt = request.form.get("next", "")
    if not username or not password:
        flash("Enter your username and password.", "error")
        return redirect(url_for("auth", next=nxt))
    user = User.query.filter((User.username == username) | (User.email == username.lower())).first()
    if not user or not bcrypt.checkpw(password.encode(), user.password):
        flash("That username and password don't match.", "error")
        return redirect(url_for("auth", next=nxt))
    session.clear()
    session["uid"] = user.id
    flash(f"Welcome back, {user.username}.", "success")
    return redirect(_safe_next(nxt))


@app.post("/register")
def register():
    f = request.form
    username, email = (f.get("username") or "").strip(), (f.get("email") or "").strip().lower()
    pw, confirm = f.get("password") or "", f.get("confirm") or ""
    back = redirect(url_for("auth", mode="register", next=f.get("next", "")))
    if not username or not email or not pw:
        flash("Fill in every field.", "error")
        return back
    if len(username) < 3 or len(username) > 64:
        flash("Username must be 3–64 characters.", "error")
        return back
    if "@" not in email or "." not in email.split("@")[-1]:
        flash("Enter a valid email address.", "error")
        return back
    if len(pw) < 6:
        flash("Password needs at least 6 characters.", "error")
        return back
    if pw != confirm:
        flash("The two passwords don't match.", "error")
        return back
    try:
        user = User(username=username, email=email, password=bcrypt.hashpw(pw.encode(), bcrypt.gensalt()))
        db.session.add(user)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        flash("That username or email is already registered.", "error")
        return back
    session.clear()
    session["uid"] = user.id
    flash(f"Account created. Welcome, {user.username}.", "success")
    return redirect(_safe_next(f.get("next", "")))


@app.route("/logout")
def logout():
    session.clear()
    flash("You're logged out.", "info")
    return redirect(url_for("index"))

# ── Old URLs from the first version ────────────────────────────────


@app.route("/predict")
@app.route("/output", methods=["GET", "POST"])
def _old_predict():
    return redirect(url_for("analyse"))


@app.route("/model")
@app.route("/result")
def _old_models():
    return redirect(url_for("models"))


@app.route("/login", methods=["GET"])
def _old_login():
    return redirect(url_for("auth"))


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", 5000)))
