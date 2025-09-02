import os
from typing import Dict, Any

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd

from utils import (
    load_dataset, discover_models,
    ensure_user_dirs, json_path_for, load_existing_annotations,
    save_annotations, sanitize_username,
    build_inline_stream, extract_tags, build_payload_inline, count_tags_in_text
)

# ---- Configuration ----
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Final_Cues_Analysis_Dataset.xlsx")
ANNOTATIONS_BASE = os.path.join(BASE_DIR, "annotations")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Load dataset once at startup
DF: pd.DataFrame = load_dataset(DATA_PATH)
MODELS = discover_models(DF)

def current_user_dir() -> str:
    username = session.get("username")
    if not username:
        return ""
    return ensure_user_dirs(ANNOTATIONS_BASE, username, MODELS)

@app.route("/")
def root():
    if not session.get("username"):
        return redirect(url_for("login"))
    return redirect(url_for("index"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Please enter your name to sign in.", "error")
            return render_template("login.html")
        session["username"] = sanitize_username(name)
        ensure_user_dirs(ANNOTATIONS_BASE, session["username"], MODELS)
        flash(f"Signed in as {session['username']}.")
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.")
    return redirect(url_for("login"))

@app.route("/index")
def index():
    if not session.get("username"):
        return redirect(url_for("login"))

    model = request.args.get("model") or (MODELS[0] if MODELS else None)
    user_dir = current_user_dir()

    rows = []
    for i in range(len(DF)):
        path = json_path_for(user_dir, model, i) if model else None
        existing = load_existing_annotations(user_dir, model, i) if (model and os.path.exists(path)) else {}
        df_row = DF.iloc[i]
        tagged_text = df_row.get(model) or df_row.get("gpt4o_layer2_annotations") or df_row.get("English Translation") or ""
        total = count_tags_in_text(tagged_text)
        decided = sum(1 for it in existing.get("items", []) if it.get("decision") in {"agree", "disagree"}) if existing else 0
        rows.append({
            "idx": i,
            "display_no": i + 1,
            "title": df_row.get("Title", ""),
            "stance": df_row.get("Stance", ""),
            "progress": f"{decided}/{total}",
        })

    return render_template(
        "index.html",
        models=MODELS,
        model=model,
        rows=rows,
        username=session.get("username"),
    )

@app.route("/annotate/<int:row_idx>", methods=["GET", "POST"])
def annotate(row_idx: int):
    if not session.get("username"):
        return redirect(url_for("login"))

    model = request.args.get("model") or request.form.get("model") or (MODELS[0] if MODELS else None)
    model = model if model in MODELS else (MODELS[0] if MODELS else None)

    df_row = DF.iloc[row_idx]
    tagged_text = df_row.get(model) or df_row.get("gpt4o_layer2_annotations") or df_row.get("English Translation") or ""

    if request.method == "POST":
        tags = extract_tags(tagged_text)
        notes = request.form.get("notes", "")
        payload = build_payload_inline(
            username=session.get("username"),
            model=model,
            row_index=row_idx,
            transcript_no=df_row.get("Transcript No", row_idx + 1),
            df_row=df_row,
            tags=tags,
            form_data=request.form,
            notes=notes,
        )
        save_annotations(current_user_dir(), model, row_idx, payload)

        intent = request.form.get("intent")
        if intent == "prev" and row_idx > 0:
            return redirect(url_for("annotate", row_idx=row_idx - 1, model=model))
        elif intent == "next" and row_idx < len(DF) - 1:
            return redirect(url_for("annotate", row_idx=row_idx + 1, model=model))
        elif intent == "index":
            return redirect(url_for("index", model=model))
        elif intent == "switch_model":
            new_model = request.form.get("model") or model
            return redirect(url_for("annotate", row_idx=row_idx, model=new_model))
        return redirect(url_for("annotate", row_idx=row_idx, model=model))

    # GET
    stream = build_inline_stream(tagged_text)
    existing = load_existing_annotations(current_user_dir(), model, row_idx)
    existing_map: Dict[int, str] = {int(item.get("tag_index")): item.get("decision") for item in existing.get("items", [])}
    notes = existing.get("notes", "")

    ctx: Dict[str, Any] = {
        "username": session.get("username"),
        "models": MODELS,
        "model": model,
        "row_idx": row_idx,
        "display_no": row_idx + 1,
        "title": df_row.get("Title", ""),
        "stance": df_row.get("Stance", ""),
        "stream": stream,  # [{kind:'text',text:'...'} | {kind:'tag',text:'Call to Action',idx:1}]
        "choices": existing_map,
        "notes": notes,
        "has_prev": row_idx > 0,
        "has_next": row_idx < len(DF) - 1,
        "total_rows": len(DF),
    }

    return render_template("annotate.html", **ctx)

if __name__ == "__main__":
    os.makedirs(ANNOTATIONS_BASE, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
