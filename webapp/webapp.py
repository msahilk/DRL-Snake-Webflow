from flask import Flask, request, session, redirect, url_for, render_template_string
import os, time, re
from markupsafe import escape

app = Flask(__name__)
app.secret_key = "devkey"

FAULT_DELAY = float(os.getenv("FAULT_DELAY_SEC", "0"))   # e.g., 0, 0.5, 1.0

BASE = """
<!doctype html><html><head><title>{{title}}</title>
<style>body{font-family:system-ui;margin:40px} .err{color:#b00;margin:6px 0}
label{display:block;margin:10px 0 2px} input{padding:6px 8px;width:280px}
button{margin-top:14px;padding:8px 12px} .nav{margin-top:20px}
small{color:#555}</style></head><body>
<h2>{{title}}</h2>
{% if error %}<div class="err">{{error}}</div>{% endif %}
{{content|safe}}
<div class="nav"><a href="{{ url_for('home') }}">Home</a> | <a href="{{ url_for('reset') }}">Reset</a></div>
<hr><small>Faults: delay={{FAULT_DELAY}}s weak_valid={{FAULT_WEAK_VALIDATION}} mislabel={{FAULT_MISLABEL}} email500={{FAULT_EMAIL_500}}</small>
</body></html>
"""

def maybe_delay():
    if FAULT_DELAY > 0:
        time.sleep(FAULT_DELAY)

@app.route("/")
def home():
    maybe_delay()
    start_href = url_for("signup1")  # <-- build URL in Python
    content = f"""
<p>Welcome. This is a 2-step signup flow.</p>
<ol>
<li>Enter first/last name.</li>
<li>Enter email and password, then submit.</li>
</ol>
<p><a id="start" href="{start_href}">Start signup</a></p>
"""
    return render_template_string(
        BASE, title="Welcome", content=content, error=None,
        FAULT_DELAY=FAULT_DELAY
    )

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("home"))

@app.route("/signup1", methods=["GET","POST"])
def signup1():
    maybe_delay()
    error = None
    if request.method == "POST":
        first = request.form.get("first", "").strip()
        last  = request.form.get("last", "").strip()
        if not first or not last:
            error = "First and last name are required."
        elif not re.match(r"^[A-Za-z' -]{2,}$", first) or not re.match(r"^[A-Za-z' -]{2,}$", last):
            error = "Names must be alphabetic and at least 2 chars."
        if error is None:
            session["first"] = first
            session["last"]  = last
            return redirect(url_for("signup2"))

    # Prefill with session values (escape for safety)
    first_prefill = escape(session.get("first",""))
    last_prefill  = escape(session.get("last",""))
    content = f"""
<form id="form1" method="post">
  <label for="first">First name</label>
  <input id="first" name="first" value="{first_prefill}" />
  <label for="last">Last name</label>
  <input id="last" name="last" value="{last_prefill}" />
  <button id="next1" type="submit">Next</button>
</form>
"""
    return render_template_string(
        BASE, title="Signup — Step 1", content=content, error=error,
        FAULT_DELAY=FAULT_DELAY
    )

@app.route("/signup2", methods=["GET","POST"])
def signup2():
    maybe_delay()
    if request.method == "POST":
        email = request.form.get("email","").strip()
        pwd = request.form.get("password","")


        if "@" not in email or "." not in email.split("@")[-1]:
            error = "Invalid email format."
            content = form2(email, pwd)
            return render_template_string(
                BASE, title="Signup — Step 2", content=content, error=error,
                FAULT_DELAY=FAULT_DELAY
            )
        if len(pwd) < 6:
            error = "Password must be >= 6 chars."
            content = form2(email, pwd)
            return render_template_string(
                BASE, title="Signup — Step 2", content=content, error=error,
                FAULT_DELAY=FAULT_DELAY
            )

        session["email"] = email
        session["password"] = pwd
        return redirect(url_for("confirm"))

    content = form2(session.get("email",""), "")
    return render_template_string(
        BASE, title="Signup — Step 2", content=content, error=None,
        FAULT_DELAY=FAULT_DELAY
    )

def form2(email, pwd):
    return f"""
<form id="form2" method="post">
  <label for="email">Email</label>
  <input id="email" name="email" value="{escape(email)}" />
  <label for="password">Password</label>
  <input id="password" name="password" type="password" value="{escape(pwd)}" />
  <button id="submit" type="submit">Submit</button>
</form>
"""

@app.route("/confirm")
def confirm():
    maybe_delay()
    first = session.get("first","")
    last  = session.get("last","")
    email = session.get("email","")

    show_first, show_last =  (first, last)

    content = f"""
<p>Thanks for signing up.</p>
<table>
<tr><td>First:</td><td id="cf_first">{escape(show_first)}</td></tr>
<tr><td>Last:</td><td id="cf_last">{escape(show_last)}</td></tr>
<tr><td>Email:</td><td id="cf_email">{escape(email)}</td></tr>
</table>
<a id="finish" href="{url_for('home')}">Finish</a>
"""
    error = None
    return render_template_string(
        BASE, title="Confirm", content=content, error=error,
        FAULT_DELAY=FAULT_DELAY
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
