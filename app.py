import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import json
import re
import hashlib
import secrets
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from supabase import create_client

# =========================================================
# SUPABASE CONNECTION
# =========================================================

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_SECRET_KEY"]
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MPLAD-AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto"
)


# =========================================================
# LOAD DATA
# =========================================================

work = joblib.load("mplad_work_clean.pkl")

scaler = joblib.load("mplad_scaler.pkl")

model = joblib.load("mplad_isolation_forest.pkl")

category_medians = joblib.load(
    "mplad_category_medians.pkl"
)

state_medians = joblib.load(
    "mplad_state_medians.pkl"
)

state_delay_medians = joblib.load(
    "mplad_state_delay_medians.pkl"
)


# =========================================================
# VERIFICATION DATABASE
# =========================================================

# Always store/read the SQLite database beside app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mplad_verification.db")


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def initialize_verification_db():
    connection = get_db_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            remarks TEXT,
            checklist TEXT NOT NULL,
            submitted_at TEXT NOT NULL
        )
        """
    )

    # Officer accounts are stored securely in the same local SQLite database.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS officer_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            officer_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            approval_status TEXT NOT NULL DEFAULT 'Pending',
            approved_at TEXT,
            approved_by TEXT
        )
        """
    )

    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(officer_users)").fetchall()
    }

    if "approval_status" not in existing_columns:
        connection.execute(
            "ALTER TABLE officer_users ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'Pending'"
        )
    if "approved_at" not in existing_columns:
        connection.execute("ALTER TABLE officer_users ADD COLUMN approved_at TEXT")
    if "approved_by" not in existing_columns:
        connection.execute("ALTER TABLE officer_users ADD COLUMN approved_by TEXT")

    connection.commit()
    connection.close()


def hash_password(password, salt=None):
    """Create a salted PBKDF2 password hash using Python's standard library."""
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000
    ).hex()

    return password_hash, salt


def verify_password(password, stored_hash, salt):
    """Check a password against the stored PBKDF2 hash."""
    password_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(password_hash, stored_hash)


def normalize_login_value(value):
    """Normalize email or phone input before authentication."""
    return str(value).strip().lower()


def authenticate_officer(login_value, password):
    """Authenticate an officer using either email or phone number."""
    login_value = normalize_login_value(login_value)

    connection = get_db_connection()

    officer = connection.execute(
        """
        SELECT id, officer_name, email, phone, password_hash, salt, approval_status
        FROM officer_users
        WHERE lower(email) = ? OR phone = ?
        """,
        (login_value, login_value)
    ).fetchone()

    connection.close()

    if officer is None:
        return None

    if not verify_password(password, officer[4], officer[5]):
        return None

    if officer[6] != "Approved":
        return {
            "id": officer[0],
            "name": officer[1],
            "email": officer[2],
            "phone": officer[3],
            "approval_status": officer[6],
            "not_approved": True,
        }

    return {
        "id": officer[0],
        "name": officer[1],
        "email": officer[2],
        "phone": officer[3],
        "approval_status": officer[6],
        "not_approved": False,
    }


def find_officer(login_value):
    """Find an officer account by email or phone."""
    login_value = normalize_login_value(login_value)

    connection = get_db_connection()

    officer = connection.execute(
        """
        SELECT id, officer_name, email, phone
        FROM officer_users
        WHERE lower(email) = ? OR phone = ?
        """,
        (login_value, login_value)
    ).fetchone()

    connection.close()
    return officer


def update_officer_password(login_value, new_password):
    """Replace an officer password with a newly generated salted hash."""
    password_hash, salt = hash_password(new_password)

    connection = get_db_connection()

    connection.execute(
        """
        UPDATE officer_users
        SET password_hash = ?, salt = ?
        WHERE lower(email) = ? OR phone = ?
        """,
        (password_hash, salt, normalize_login_value(login_value), normalize_login_value(login_value))
    )

    connection.commit()
    connection.close()


def send_password_reset_otp(email_address):
    """Generate and email a short-lived password-reset OTP."""
    smtp_host = os.getenv("MPLAD_SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("MPLAD_SMTP_PORT", "587"))
    smtp_user = os.getenv("MPLAD_SMTP_USER", "").strip()
    smtp_password = os.getenv("MPLAD_SMTP_PASSWORD", "")
    from_email = os.getenv("MPLAD_SMTP_FROM", smtp_user).strip()
    use_tls = os.getenv("MPLAD_SMTP_TLS", "true").strip().lower() != "false"

    if not all([smtp_host, smtp_user, smtp_password, from_email]):
        return False, "Email OTP is not configured. Add the MPLAD SMTP environment variables first."

    otp = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = datetime.now() + timedelta(minutes=5)

    message = EmailMessage()
    message["Subject"] = "MPLAD-AI Password Reset OTP"
    message["From"] = from_email
    message["To"] = email_address
    message.set_content(
        f"Your MPLAD-AI password reset OTP is {otp}.\n\n"
        "This OTP is valid for 5 minutes. If you did not request a password reset, ignore this email."
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)
    except Exception as exc:
        return False, f"Unable to send the OTP email: {exc}"

    st.session_state["password_reset_otp"] = otp
    st.session_state["password_reset_expires"] = expires_at
    return True, None


def is_valid_password_reset_otp(otp):
    """Validate the emailed OTP and its five-minute expiry."""
    stored_otp = st.session_state.get("password_reset_otp")
    expires_at = st.session_state.get("password_reset_expires")

    if not stored_otp or not expires_at:
        return False

    if datetime.now() > expires_at:
        return False

    return secrets.compare_digest(str(otp).strip(), stored_otp)


def clear_password_reset_state():
    """Clear all temporary password-recovery values."""
    for key in [
        "password_reset_login",
        "password_reset_email",
        "password_reset_otp",
        "password_reset_expires",
        "password_reset_step",
    ]:
        st.session_state.pop(key, None)


def create_officer_account(officer_name, email, phone, password):
    """Create a new officer account with a salted password hash."""
    email = normalize_login_value(email)
    phone = re.sub(r"\\D", "", str(phone).strip())

    if not officer_name.strip():
        return False, "Officer name is required."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Enter a valid email address."
    if not re.fullmatch(r"\d{10}", phone):
        return False, "Enter a valid 10-digit mobile number."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash, salt = hash_password(password)

    connection = get_db_connection()
    try:
        connection.execute(
            """
            INSERT INTO officer_users
            (officer_name, email, phone, password_hash, salt, created_at, approval_status)
            VALUES (?, ?, ?, ?, ?, ?, 'Pending')
            """,
            (
                officer_name.strip(),
                email,
                phone,
                password_hash,
                salt,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()
    except sqlite3.IntegrityError:
        return False, "An officer account with this email or phone already exists."
    finally:
        connection.close()

    return True, None


def get_admin_credentials():
    """Read administrator credentials from Streamlit secrets or environment variables."""
    try:
        admin_email = str(st.secrets.get("MPLAD_ADMIN_EMAIL", "")).strip().lower()
        admin_password = str(st.secrets.get("MPLAD_ADMIN_PASSWORD", ""))
    except Exception:
        admin_email = ""
        admin_password = ""

    admin_email = os.getenv("MPLAD_ADMIN_EMAIL", admin_email).strip().lower()
    admin_password = os.getenv("MPLAD_ADMIN_PASSWORD", admin_password)
    return admin_email, admin_password


def authenticate_admin(email, password):
    """Authenticate administrator credentials stored outside the application database."""
    admin_email, admin_password = get_admin_credentials()
    return bool(
        admin_email
        and admin_password
        and secrets.compare_digest(str(email).strip().lower(), admin_email)
        and secrets.compare_digest(str(password), admin_password)
    )


def get_pending_officers():
    connection = get_db_connection()
    rows = connection.execute(
        """
        SELECT id, officer_name, email, phone, created_at, approval_status
        FROM officer_users
        WHERE approval_status = 'Pending'
        ORDER BY id DESC
        """
    ).fetchall()
    connection.close()
    return rows


def get_all_officers():
    connection = get_db_connection()
    rows = connection.execute(
        """
        SELECT id, officer_name, email, phone, created_at,
               approval_status, approved_at, approved_by
        FROM officer_users
        ORDER BY id DESC
        """
    ).fetchall()
    connection.close()
    return rows


def update_officer_approval(officer_id, status, admin_email):
    connection = get_db_connection()
    decision_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    connection.execute(
        """
        UPDATE officer_users
        SET approval_status = ?, approved_at = ?, approved_by = ?
        WHERE id = ?
        """,
        (status, decision_time, admin_email, int(officer_id)),
    )

    connection.commit()
    connection.close()


def save_verification(
    work_id,
    status,
    remarks,
    checklist
):
    connection = get_db_connection()

    connection.execute(
        """
        INSERT INTO verification_records
        (work_id, status, remarks, checklist, submitted_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(work_id),
            status,
            remarks,
            json.dumps(checklist),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()


def get_verification_history(work_id=None):
    connection = get_db_connection()

    if work_id is None:
        query = """
            SELECT id, work_id, status, remarks, checklist, submitted_at
            FROM verification_records
            ORDER BY id DESC
        """
        data = pd.read_sql_query(query, connection)
    else:
        query = """
            SELECT id, work_id, status, remarks, checklist, submitted_at
            FROM verification_records
            WHERE work_id = ?
            ORDER BY id DESC
        """
        data = pd.read_sql_query(
            query,
            connection,
            params=(int(work_id),)
        )

    connection.close()
    return data


initialize_verification_db()


# =========================================================
# AI-GUIDED VERIFICATION CHECKLIST
# =========================================================

def get_verification_checklist(row):
    """
    Generate a focused verification checklist for the selected work.

    The checklist is based on:
    1. Detected cost-related risk
    2. Detected delay-related risk
    3. Current work stage

    The AI recommends what the officer should check. It does not
    automatically verify any item.
    """

    checklist = []

    # -----------------------------------------------------
    # Always verify the identity/scope of the work.
    # -----------------------------------------------------
    checklist.append(
        "Work description matches the official sanctioned record"
    )

    # -----------------------------------------------------
    # COST-FOCUSED CHECKS
    # Only add these when the AI sees a meaningful cost deviation.
    # -----------------------------------------------------
    cost_risk = (
        row["COST_DEVIATION"] >= 5
        or row["STATE_COST_DEVIATION"] >= 3
    )

    if cost_risk:
        checklist.extend([
            "Sanction amount matches the official sanction record",
            "Work scope is consistent with the sanctioned amount",
            "Supporting cost/sanction documents are checked, where available"
        ])
    else:
        # For normal-cost works, one basic financial check is enough.
        checklist.append(
            "Sanction amount matches the official sanction record"
        )

    # -----------------------------------------------------
    # DELAY-FOCUSED CHECKS
    # Only add these when the AI sees a meaningful delay pattern.
    # -----------------------------------------------------
    delay_risk = (
        row["SANCTION_DELAY_DAYS"] >= 300
        or row["STATE_DELAY_DEVIATION"] >= 1.5
    )

    if delay_risk:
        checklist.extend([
            "Recommendation date and sanction date are verified",
            "Sanction delay is verified against the official record",
            "Reason or justification for the unusual delay is checked"
        ])

    # -----------------------------------------------------
    # WORK-STAGE CHECKS
    # Only the checks relevant to the current stage are added.
    # -----------------------------------------------------
    stage = str(row["WORK_STAGE"]).strip().lower()

    if "physical inspection" in stage:
        checklist.extend([
            "Physical work/progress is checked against available evidence",
            "Photographs or inspection evidence are checked, where available"
        ])

    elif "vendor identification" in stage:
        checklist.extend([
            "Implementing Agency/vendor information is checked",
            "Relevant vendor/agency supporting documents are checked, where available"
        ])

    elif "partially completed" in stage:
        checklist.extend([
            "Current work progress is checked against the official record",
            "Payment/expenditure information is checked, where applicable",
            "Available progress photographs/evidence are checked"
        ])

    elif "completed" in stage:
        checklist.extend([
            "Completion status is verified",
            "Completion evidence/photographs are checked, where available",
            "Final payment/expenditure information is checked, where applicable"
        ])

    elif "sanction" in stage:
        checklist.append(
            "Sanction details are verified against the official record"
        )

    elif "time estimation" in stage:
        checklist.append(
            "Estimated timeline is checked against the work requirements"
        )

    # Remove duplicate checks while preserving their order.
    return list(dict.fromkeys(checklist))


def get_verification_source(check_item):
    """Return the official source the officer should use for a checklist item."""

    if "Work description" in check_item:
        return "Official eSAKSHI work/sanction record"

    if "Sanction amount" in check_item:
        return "eSAKSHI sanction details / official sanction order"

    if "Work scope" in check_item:
        return "Official sanction order and sanctioned work details"

    if "cost/sanction documents" in check_item:
        return "eSAKSHI uploaded documents / official sanction records"

    if "Recommendation date" in check_item:
        return "eSAKSHI recommendation and sanction records"

    if "Sanction delay" in check_item:
        return "Recommendation and sanction dates in eSAKSHI"

    if "justification" in check_item:
        return "Official sanction record and supporting justification/documents"

    if "Physical work/progress" in check_item:
        return "eSAKSHI work-stage/progress record and inspection evidence"

    if "Photographs" in check_item or "photographs" in check_item:
        return "Photographs / inspection evidence available in the official record"

    if "Implementing Agency/vendor" in check_item:
        return "eSAKSHI Implementing Agency/vendor information"

    if "vendor/agency" in check_item:
        return "eSAKSHI uploaded vendor/agency supporting documents"

    if "Current work progress" in check_item:
        return "eSAKSHI current work-stage/progress record"

    if "Payment/expenditure" in check_item or "payment/expenditure" in check_item:
        return "eSAKSHI payment/expenditure information"

    if "Completion status" in check_item:
        return "eSAKSHI completion/work-stage record"

    if "Completion evidence" in check_item:
        return "eSAKSHI completion evidence / photographs"

    if "Final payment" in check_item:
        return "eSAKSHI final payment/expenditure information"

    if "Sanction details" in check_item:
        return "eSAKSHI sanction record / sanction order"

    if "Estimated timeline" in check_item:
        return "Official work record and sanction/timeline documents"

    if "Available progress" in check_item:
        return "eSAKSHI progress photographs/evidence"

    return "Official eSAKSHI record or supporting document"


# =========================================================
# MODERN CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MPLAD-AI MODERN 3D UI SYSTEM
       Visual-only changes: application logic is untouched.
       ===================================================== */

    :root {
        --mplad-bg: #f4f7fb;
        --mplad-card: rgba(255, 255, 255, 0.92);
        --mplad-border: rgba(148, 163, 184, 0.20);
        --mplad-text: #0f172a;
        --mplad-muted: #64748b;
        --mplad-primary: #2563eb;
        --mplad-primary-dark: #1d4ed8;
        --mplad-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
        --mplad-shadow-hover: 0 20px 45px rgba(15, 23, 42, 0.14);
    }

    /* Hide Streamlit branding elements while keeping the header
       available so the mobile sidebar control remains usable. */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: visible;
        background: transparent;
    }

    /* App background */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 5%, rgba(37, 99, 235, 0.08), transparent 28%),
            radial-gradient(circle at 92% 8%, rgba(99, 102, 241, 0.07), transparent 25%),
            linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    [data-testid="stMainBlockContainer"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
        max-width: 1500px;
    }

    /* =====================================================
       SIDEBAR - GLASS / 3D COMMAND CENTER
       ===================================================== */

    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 18% 8%, rgba(59,130,246,0.18), transparent 26%),
            radial-gradient(circle at 85% 35%, rgba(99,102,241,0.14), transparent 30%),
            linear-gradient(180deg, #0b1220 0%, #101827 52%, #111c3b 100%);
        border-right: 1px solid rgba(255,255,255,0.10);
        box-shadow: 14px 0 42px rgba(2,6,23,0.24);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.15rem 0.9rem 1rem 0.9rem;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    /* Decorative glow behind the sidebar */
    [data-testid="stSidebar"]::before {
        content: "";
        position: fixed;
        top: 74px;
        left: 28px;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: rgba(59,130,246,0.10);
        filter: blur(28px);
        pointer-events: none;
    }

    /* Glass brand card */
    .mplad-sidebar-brand {
        position: relative;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px;
        margin: 2px 2px 18px 2px;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255,255,255,0.11), rgba(255,255,255,0.045));
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.10),
            0 14px 30px rgba(0,0,0,0.18);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .mplad-sidebar-orb {
        width: 42px;
        height: 42px;
        min-width: 42px;
        display: grid;
        place-items: center;
        border-radius: 14px;
        font-size: 19px;
        font-weight: 900;
        background: linear-gradient(145deg, #3b82f6, #6366f1);
        box-shadow:
            0 9px 22px rgba(59,130,246,0.34),
            inset 0 1px 1px rgba(255,255,255,0.45);
    }

    .mplad-sidebar-title {
        font-size: 1.05rem;
        font-weight: 850;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }

    .mplad-sidebar-subtitle {
        margin-top: 4px;
        color: rgba(226,232,240,0.68) !important;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .mplad-sidebar-section {
        margin: 4px 5px 8px 5px;
        color: rgba(226,232,240,0.58) !important;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* =====================================================
       SIDEBAR NAVIGATION - CLEAR GLASS HOVER / PRESS
       Uses high-specificity selectors so Streamlit theme CSS
       cannot hide the hover state.
       ===================================================== */

    [data-testid="stSidebar"] [data-testid="stButton"] {
        width: 100%;
        margin: 0 0 10px 0;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button,
    [data-testid="stSidebar"] div.stButton > button {
        position: relative !important;
        width: 100% !important;
        min-height: 48px !important;
        padding: 0.68rem 0.95rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.035) !important;
        background: rgba(255,255,255,0.025) !important;
        background-image: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.008)) !important;
        color: #e8edf7 !important;
        font-size: 0.91rem !important;
        font-weight: 750 !important;
        text-align: left !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.07),
            inset 0 -1px 0 rgba(0,0,0,0.16),
            0 3px 0 rgba(3,10,25,0.42),
            0 9px 18px rgba(0,0,0,0.16) !important;
        overflow: hidden !important;
        transform: translateY(0) translateZ(0) !important;
        transform-style: preserve-3d !important;
        filter: none !important;
        transition:
            transform 160ms cubic-bezier(.2,.8,.2,1),
            background 160ms ease,
            border-color 160ms ease,
            box-shadow 160ms ease,
            color 160ms ease !important;
        -webkit-tap-highlight-color: transparent !important;
    }

    /* Bright glass edge that appears only on hover. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button::before {
        content: "" !important;
        position: absolute !important;
        inset: 0 !important;
        border-radius: inherit !important;
        border: 1px solid transparent !important;
        background: linear-gradient(135deg, rgba(96,165,250,0.80), rgba(129,140,248,0.55)) border-box !important;
        -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0) border-box !important;
        -webkit-mask-composite: xor !important;
        mask-composite: exclude !important;
        opacity: 0 !important;
        transform: scale(0.985) !important;
        transition: opacity 160ms ease, transform 160ms ease !important;
        pointer-events: none !important;
        z-index: 0 !important;
    }

    /* Moving glass highlight. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button::after {
        content: "" !important;
        position: absolute !important;
        top: 0 !important;
        bottom: 0 !important;
        left: -75% !important;
        width: 45% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent) !important;
        transform: skewX(-18deg) translateX(0) !important;
        opacity: 0 !important;
        transition: transform 420ms ease, opacity 120ms ease !important;
        pointer-events: none !important;
        z-index: 1 !important;
    }

    /* PHYSICAL HOVER: the whole navigation button visibly rises when the pointer touches it, with a tactile lower shadow. */
    [data-testid="stSidebar"] [data-testid="stButton"]:hover > button,
    [data-testid="stSidebar"] div.stButton:hover > button,
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        transform: translate3d(0, -6px, 0) scale(1.018) !important;
        background: linear-gradient(135deg, rgba(59,130,246,0.34), rgba(79,70,229,0.25)) !important;
        border-color: rgba(96,165,250,0.82) !important;
        color: #ffffff !important;
        box-shadow:
            0 3px 0 rgba(30,64,175,0.48),
            0 10px 0 rgba(15,23,42,0.18),
            0 22px 38px rgba(0,0,0,0.38),
            0 0 0 1px rgba(96,165,250,0.25),
            0 0 32px rgba(59,130,246,0.38),
            inset 0 1px 0 rgba(255,255,255,0.24),
            inset 0 -1px 0 rgba(0,0,0,0.10) !important;
        will-change: transform, box-shadow !important;
    }

    /* Keep the raised/tactile state when a touch device focuses the button. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible,
    [data-testid="stSidebar"] [data-testid="stButton"] > button:focus {
        transform: translate3d(0, -7px, 0) scale(1.02) !important;
        background: linear-gradient(135deg, rgba(59,130,246,0.32), rgba(79,70,229,0.24)) !important;
        border-color: rgba(96,165,250,0.75) !important;
        box-shadow:
            0 8px 0 rgba(15,23,42,0.16),
            0 18px 32px rgba(0,0,0,0.30),
            0 0 24px rgba(59,130,246,0.30) !important;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover::before {
        opacity: 1 !important;
        transform: scale(1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover::after {
        opacity: 1 !important;
        transform: skewX(-18deg) translateX(390%) !important;
    }

    /* Physical button press. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button:active,
    [data-testid="stSidebar"] div.stButton > button:active {
        transform: translateY(-1px) scale(0.985) translateZ(0) !important;
        background: linear-gradient(135deg, rgba(37,99,235,0.38), rgba(79,70,229,0.29)) !important;
        border-color: rgba(147,197,253,0.75) !important;
        box-shadow:
            inset 0 5px 12px rgba(2,6,23,0.34),
            inset 0 -1px 0 rgba(255,255,255,0.05),
            0 1px 2px rgba(2,6,23,0.30) !important;
        transition-duration: 60ms !important;
    }

    /* Keyboard/touch focus. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible {
        outline: none !important;
        border-color: rgba(147,197,253,0.80) !important;
        box-shadow:
            0 0 0 3px rgba(59,130,246,0.20),
            0 10px 24px rgba(2,6,23,0.24) !important;
    }

    /* Active page: same glass language, but persistent. */
    /* ACTIVE / SELECTED: raised tactile card like the reference design.
       The selected workspace item visibly sits above the sidebar surface. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
        transform: translate3d(0, -3px, 0) scale(1.01) !important;
        background: linear-gradient(135deg, rgba(43,92,171,0.62), rgba(55,56,150,0.58)) !important;
        border: 2px solid rgba(96,165,250,0.78) !important;
        color: #ffffff !important;
        box-shadow:
            0 3px 0 rgba(18,43,94,0.78),
            0 8px 0 rgba(3,10,25,0.20),
            0 16px 28px rgba(0,0,0,0.30),
            0 0 0 2px rgba(59,130,246,0.12),
            0 0 22px rgba(59,130,246,0.28),
            inset 0 1px 0 rgba(255,255,255,0.24),
            inset 0 -2px 0 rgba(8,20,48,0.18) !important;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]::before {
        opacity: 1 !important;
        transform: scale(1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translate3d(0, -7px, 0) scale(1.02) !important;
        background: linear-gradient(135deg, rgba(53,105,194,0.72), rgba(62,61,165,0.66)) !important;
        border-color: rgba(147,197,253,0.96) !important;
        box-shadow:
            0 4px 0 rgba(18,43,94,0.82),
            0 12px 0 rgba(3,10,25,0.22),
            0 24px 38px rgba(0,0,0,0.36),
            0 0 0 2px rgba(96,165,250,0.18),
            0 0 30px rgba(59,130,246,0.38),
            inset 0 1px 0 rgba(255,255,255,0.28),
            inset 0 -2px 0 rgba(8,20,48,0.18) !important;
    }

    /* When the selected item is physically pressed, it sinks into the surface. */
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:active {
        transform: translate3d(0, 1px, 0) scale(0.985) !important;
        border-color: rgba(191,219,254,0.98) !important;
        box-shadow:
            inset 0 5px 13px rgba(2,6,23,0.42),
            inset 0 -1px 0 rgba(255,255,255,0.08),
            0 1px 2px rgba(2,6,23,0.30),
            0 0 10px rgba(59,130,246,0.20) !important;
        transition-duration: 55ms !important;
    }

    /* Sidebar system status card removed. */

    /* Headings */
    h1, h2, h3 {
        color: var(--mplad-text) !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 750 !important;
    }

    p {
        color: #475569;
    }

    /* Metric cards - elevated 3D/glass effect */
    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(248,250,255,0.90));
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 18px;
        padding: 20px;
        box-shadow:
            0 10px 25px rgba(15,23,42,0.07),
            inset 0 1px 0 rgba(255,255,255,0.90);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        cursor: pointer;
        transform: translateY(0) scale(1);
        will-change: transform, box-shadow;
        transition:
            transform 0.18s cubic-bezier(.2,.8,.2,1),
            box-shadow 0.18s ease,
            border-color 0.18s ease,
            background 0.18s ease;
    }

    [data-testid="stMetric"] {
        border-top: 3px solid #4f7cff;
    }

    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -38px;
        bottom: -42px;
        border-radius: 50%;
        background: rgba(37, 99, 235, 0.07);
        pointer-events: none;
    }

    [data-testid="stMetric"]::marker {
        display: none;
    }

    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 70%;
        height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,0.42), transparent);
        transform: skewX(-18deg);
        transition: left 0.55s ease;
        pointer-events: none;
        z-index: 0;
    }

    /* Physical 3D hover — same raised interaction language as Workspace navigation */
    [data-testid="stMetric"]:hover {
        transform: translate3d(0, -9px, 0) scale(1.025);
        box-shadow:
            0 4px 0 rgba(37,99,235,0.14),
            0 12px 0 rgba(15,23,42,0.06),
            0 24px 42px rgba(15,23,42,0.18),
            0 0 0 1px rgba(96,165,250,0.18),
            0 0 30px rgba(59,130,246,0.22),
            inset 0 1px 0 rgba(255,255,255,1);
        border-color: rgba(96,165,250,0.68);
        background: linear-gradient(145deg, rgba(255,255,255,1), rgba(241,246,255,0.96));
    }

    [data-testid="stMetric"]:hover::before {
        left: 135%;
    }

    [data-testid="stMetric"]:active {
        transform: translate3d(0, -2px, 0) scale(0.985);
        box-shadow:
            0 5px 12px rgba(15,23,42,0.12),
            inset 0 3px 8px rgba(15,23,42,0.08),
            inset 0 1px 0 rgba(255,255,255,0.75);
        border-color: rgba(37,99,235,0.42);
        transition-duration: 0.07s;
    }

    [data-testid="stMetric"]:active::after {
        transform: scale(1.12);
        background: rgba(37,99,235,0.11);
    }

    [data-testid="stMetric"] > div {
        position: relative;
        z-index: 1;
    }



    @media (hover: none) {
        [data-testid="stMetric"]:active {
            transform: scale(0.985);
        }
    }

    @media (hover: none) {
        [data-testid="stSidebar"] [data-testid="stButton"] > button:active,
        [data-testid="stSidebar"] div.stButton > button:active {
            transform: translateY(-1px) scale(0.97) !important;
            box-shadow: inset 0 5px 12px rgba(2,6,23,0.34), 0 1px 2px rgba(2,6,23,0.28) !important;
        }
    }

    /* =====================================================
       APP-WIDE INTERACTION / HOVER SYSTEM
       ===================================================== */

    [data-testid="stExpander"],
    [data-testid="stForm"],
    [data-testid="stAlert"],
    [data-testid="stDataFrame"] {
        transition:
            transform 0.22s cubic-bezier(.2,.8,.2,1),
            box-shadow 0.22s ease,
            border-color 0.22s ease;
    }

    [data-testid="stExpander"]:hover,
    [data-testid="stForm"]:hover,
    [data-testid="stAlert"]:hover,
    [data-testid="stDataFrame"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(15,23,42,0.10);
        border-color: rgba(37,99,235,0.20);
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        position: relative;
        overflow: hidden;
        transition:
            transform 0.20s cubic-bezier(.2,.8,.2,1),
            box-shadow 0.20s ease,
            filter 0.20s ease,
            border-color 0.20s ease;
    }

    .stButton > button::after,
    [data-testid="stFormSubmitButton"] button::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 25%, rgba(255,255,255,0.18) 50%, transparent 75%);
        transform: translateX(-120%);
        transition: transform 0.55s ease;
        pointer-events: none;
    }

    .stButton > button:hover::after,
    [data-testid="stFormSubmitButton"] button:hover::after {
        transform: translateX(120%);
    }

    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        transition:
            transform 0.18s ease,
            border-color 0.20s ease,
            box-shadow 0.20s ease,
            background 0.20s ease !important;
    }

    [data-baseweb="select"] > div:hover,
    [data-testid="stTextInput"] input:hover,
    [data-testid="stNumberInput"] input:hover,
    [data-testid="stTextArea"] textarea:hover {
        transform: translateY(-1px);
        border-color: rgba(37,99,235,0.35) !important;
        box-shadow: 0 7px 18px rgba(15,23,42,0.07) !important;
    }

    [data-testid="stProgress"] > div > div > div {
        transition: filter 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stProgress"]:hover > div > div > div {
        filter: brightness(1.06);
        box-shadow: 0 0 12px rgba(37,99,235,0.22);
    }

    [data-testid="stMetricLabel"] {
        color: var(--mplad-muted) !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--mplad-text) !important;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    /* Buttons */
    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 11px;
        font-weight: 700;
        min-height: 42px;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: white;
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.18);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.26);
        filter: brightness(1.03);
    }

    .stButton > button:active,
    [data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0);
    }

    /* Inputs / selectors */
    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        border-radius: 11px !important;
        border-color: #dbe3ee !important;
        background: rgba(255, 255, 255, 0.92) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    [data-baseweb="select"] > div:focus-within,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(37, 99, 235, 0.55) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10) !important;
    }

    /* Forms */
    [data-testid="stForm"] {
        border: 1px solid var(--mplad-border);
        border-radius: 18px;
        padding: 25px;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: var(--mplad-shadow);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* Alerts / information boxes */
    [data-testid="stAlert"] {
        border-radius: 14px;
        border-width: 1px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    /* Charts and generic containers */
    [data-testid="stExpander"] {
        border-radius: 14px;
        border-color: #e2e8f0;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    /* Progress bars */
    [data-testid="stProgress"] > div > div {
        border-radius: 999px;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        color: #64748b;
    }

    /* Section spacing */
    .section-space {
        margin-top: 25px;
    }

    /* Smooth appearance */
    .stMarkdown, [data-testid="stMetric"], [data-testid="stAlert"],
    [data-testid="stForm"], [data-testid="stDataFrame"] {
        animation: mpladFadeUp 0.35s ease both;
    }

    @keyframes mpladFadeUp {
        from {
            opacity: 0;
            transform: translateY(5px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* =====================================================
       FAST MODERN VERIFICATION SEARCH
       ===================================================== */

    .verification-search-label {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 14px 0 8px 2px;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .search-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2563eb;
        box-shadow: 0 0 0 5px rgba(37,99,235,0.10), 0 0 14px rgba(37,99,235,0.28);
    }

    .verification-search-label + div [data-testid="stForm"] {
        padding: 7px !important;
        border-radius: 17px !important;
        background: rgba(255,255,255,0.72) !important;
        box-shadow: 0 8px 24px rgba(15,23,42,0.07) !important;
    }

    [data-testid="stTextInput"] input {
        min-height: 48px !important;
        padding: 0 16px !important;
        border: 1px solid rgba(148,163,184,0.28) !important;
        border-radius: 13px !important;
        background: rgba(248,250,252,0.96) !important;
        color: #0f172a !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: rgba(37,99,235,0.58) !important;
        background: #ffffff !important;
        box-shadow: 0 0 0 4px rgba(37,99,235,0.09), 0 8px 22px rgba(37,99,235,0.08) !important;
    }

    [data-testid="stFormSubmitButton"] button {
        min-height: 48px !important;
        border-radius: 13px !important;
        font-weight: 800 !important;
    }

    /* =====================================================
       MOBILE RESPONSIVE DESIGN
       ===================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding: 1rem 0.8rem 2rem 0.8rem;
            max-width: 100%;
        }

        h1 {
            font-size: 1.9rem !important;
            line-height: 1.15 !important;
        }

        h2 {
            font-size: 1.4rem !important;
            line-height: 1.2 !important;
        }

        h3 {
            font-size: 1.15rem !important;
            line-height: 1.25 !important;
        }

        p {
            font-size: 0.94rem;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }

        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }

        [data-testid="stMetric"] {
            padding: 16px;
            border-radius: 15px;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.84rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            line-height: 1.15 !important;
        }

        [data-testid="stForm"] {
            padding: 15px;
            border-radius: 15px;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            min-height: 46px;
            width: 100%;
        }

        [data-testid="stDataFrame"] {
            width: 100% !important;
            overflow-x: auto !important;
        }

        [data-testid="stProgress"] {
            width: 100% !important;
        }

        [data-testid="stSidebar"] {
            min-width: 280px;
            max-width: 88vw;
        }

        .mplad-sidebar-brand {
            margin-bottom: 14px;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {
            overflow-wrap: anywhere;
            word-break: normal;
        }
    }

    /* Extra-small phones */
    @media (max-width: 480px) {

        .block-container {
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }

        h1 {
            font-size: 1.65rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.42rem !important;
        }

        [data-testid="stMetric"] {
            padding: 13px;
            border-radius: 13px;
        }

        [data-testid="stSidebar"] {
            min-width: 250px;
        }

        .mplad-status-value {
            font-size: 0.78rem;
        }
    }



    /* =====================================================
       MPLAD-AI ASSISTANT — SIDEBAR LAUNCHER
       ===================================================== */

    /* Keep the assistant trigger inside the sidebar empty space. */
    [data-testid="stSidebar"] [data-testid="stPopover"] {
        width: 100% !important;
        margin-top: 22px !important;
        display: flex !important;
        justify-content: center !important;
        position: relative !important;
    }

    /* Visible 3D robot launcher.
       The selector intentionally targets the button anywhere inside the popover
       instead of requiring it to be a direct child. */
    [data-testid="stSidebar"] [data-testid="stPopover"] button {
        position: relative !important;
        width: 260px !important;
        height: 158px !important;
        min-height: 158px !important;
        padding: 0 !important;
        border-radius: 24px !important;
        border: 1px solid rgba(125,211,252,0.65) !important;
        background-image:
            linear-gradient(180deg, rgba(5,15,40,0.04) 10%, rgba(5,15,40,0.78) 100%),
            url("data:image/png;base64,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") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        box-shadow:
            0 12px 26px rgba(2,8,23,0.48),
            0 0 0 1px rgba(96,165,250,0.12) inset,
            0 0 28px rgba(37,99,235,0.16) !important;
        color: transparent !important;
        overflow: hidden !important;
        transform: translateY(0) scale(1) !important;
        transition: transform .22s cubic-bezier(.2,.8,.2,1),
                    box-shadow .22s ease,
                    border-color .22s ease,
                    filter .22s ease !important;
    }

    /* Assistant title shown over the robot image. */
    [data-testid="stSidebar"] [data-testid="stPopover"] button::after {
        content: "✦  ASK MPLAD-AI" !important;
        position: absolute !important;
        left: 16px !important;
        right: 16px !important;
        bottom: 13px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 34px !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em !important;
        background: linear-gradient(135deg, rgba(37,99,235,.92), rgba(79,70,229,.92)) !important;
        border: 1px solid rgba(191,219,254,.45) !important;
        box-shadow: 0 8px 20px rgba(30,64,175,.30), inset 0 1px rgba(255,255,255,.25) !important;
        pointer-events: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
        transform: translateY(-9px) scale(1.025) !important;
        border-color: rgba(96,165,250,0.95) !important;
        box-shadow:
            0 22px 38px rgba(2,8,23,0.55),
            0 0 0 2px rgba(59,130,246,0.20) inset,
            0 0 34px rgba(59,130,246,0.42) !important;
        filter: brightness(1.06) saturate(1.08) !important;
    }

    [data-testid="stSidebar"] [data-testid="stPopover"] button:active {
        transform: translateY(-2px) scale(.985) !important;
        box-shadow:
            0 6px 12px rgba(2,8,23,0.55),
            0 0 0 2px rgba(59,130,246,0.16) inset !important;
    }

    [data-testid="stSidebar"] [data-testid="stPopover"] button:focus-visible {
        outline: 2px solid rgba(125,211,252,0.95) !important;
        outline-offset: 3px !important;
    }

    /* Popover chat window */
    [data-testid="stPopoverBody"] {
        width: min(390px, calc(100vw - 28px)) !important;
        max-height: min(650px, calc(100vh - 110px)) !important;
        overflow-y: auto !important;
        padding: 0.8rem !important;
        border: 1px solid rgba(147,197,253,0.38) !important;
        border-radius: 22px !important;
        background: rgba(248,250,252,0.96) !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 28px 70px rgba(15,23,42,0.28), 0 0 35px rgba(37,99,235,0.10) !important;
    }

    .mplad-float-chat-head {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 10px;
        padding: 12px;
        border-radius: 16px;
        color: white;
        background: linear-gradient(135deg, #101b3b, #2944a5);
        box-shadow: 0 9px 22px rgba(15,23,42,0.16);
    }

    .mplad-float-chat-avatar {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 42px;
        border-radius: 14px;
        background: linear-gradient(145deg, #638bff, #334bd1);
        border: 1px solid rgba(255,255,255,0.45);
        box-shadow: 0 7px 18px rgba(37,99,235,0.32);
        font-size: 1.35rem;
    }

    .mplad-float-chat-name {
        font-weight: 800;
        font-size: 0.98rem;
    }

    .mplad-float-chat-status {
        margin-top: 2px;
        font-size: 0.70rem;
        color: rgba(255,255,255,0.74);
    }

    .mplad-float-message {
        margin: 8px 0;
        padding: 10px 12px;
        border-radius: 15px;
        font-size: 0.82rem;
        line-height: 1.45;
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 7px 16px rgba(15,23,42,0.07);
    }

    .mplad-float-message.assistant {
        margin-right: 34px;
        background: #e9eef6;
        color: #334155;
        border-top-left-radius: 6px;
    }

    .mplad-float-message.user {
        margin-left: 34px;
        background: linear-gradient(135deg, #263b91, #3d4fd1);
        color: white;
        border-color: rgba(96,165,250,0.35);
        border-top-right-radius: 6px;
    }

    .mplad-float-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 22px rgba(15,23,42,0.12);
    }

    .mplad-quick-label {
        margin: 12px 0 7px;
        color: #475569;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }

    /* Chat controls: keep text readable against the blue glass buttons. */
    [data-testid="stPopoverBody"] button {
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 13px !important;
        border: 1px solid rgba(96,165,250,0.42) !important;
        background: linear-gradient(135deg, #2d55d7, #4b46d9) !important;
        box-shadow: 0 7px 15px rgba(37,99,235,0.18), inset 0 1px 1px rgba(255,255,255,0.16) !important;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
    }

    [data-testid="stPopoverBody"] button:hover {
        color: #ffffff !important;
        transform: translateY(-3px) !important;
        filter: brightness(1.06) !important;
        box-shadow: 0 13px 24px rgba(37,99,235,0.28), 0 0 14px rgba(59,130,246,0.18) !important;
    }

    [data-testid="stPopoverBody"] button:active {
        color: #ffffff !important;
        transform: translateY(1px) scale(0.98) !important;
        box-shadow: 0 4px 8px rgba(15,23,42,0.20), inset 0 3px 7px rgba(15,23,42,0.24) !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stTextInput"] input {
        color: #0f172a !important;
        background: rgba(255,255,255,0.96) !important;
        border: 1px solid rgba(148,163,184,0.30) !important;
        border-radius: 13px !important;
        box-shadow: inset 0 1px 2px rgba(15,23,42,0.04) !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stTextInput"] input:focus {
        border-color: rgba(59,130,246,0.72) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }

    /* Make chat bubbles feel like separate physical message cards. */
    .mplad-float-message {
        transition: transform .18s ease, box-shadow .18s ease;
    }

    .mplad-float-message:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 14px 26px rgba(15,23,42,0.14) !important;
    }

    @media (max-width: 480px) {
        [data-testid="stPopoverBody"] {
            width: min(350px, calc(100vw - 18px)) !important;
            max-height: calc(100vh - 70px) !important;
        }

        .mplad-float-message {
            font-size: 0.80rem;
        }
    }

    @media (max-width: 768px) {
        [data-testid="stPopover"] {
            right: 16px !important;
            bottom: 16px !important;
        }

        [data-testid="stSidebar"] [data-testid="stPopover"] > button {
            width: 56px !important;
            height: 56px !important;
            min-height: 56px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)



# =========================================================
# OFFICER AUTHENTICATION
# =========================================================

# The app uses a real SQLite-backed officer account and login session.
# No demo account or hard-coded credentials are created.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

if "admin_email" not in st.session_state:
    st.session_state["admin_email"] = ""

if "officer_name" not in st.session_state:
    st.session_state["officer_name"] = ""

if "officer_email" not in st.session_state:
    st.session_state["officer_email"] = ""

if "password_reset_step" not in st.session_state:
    st.session_state["password_reset_step"] = "request"

if "login_mode" not in st.session_state:
    st.session_state["login_mode"] = "login"

# Direct administrator portal URL:
# https://your-app.streamlit.app/?portal=admin
# This keeps the officer portal at the normal URL and opens the
# administrator approval portal only when the admin URL is used.
try:
    requested_portal = str(st.query_params.get("portal", "")).strip().lower()
except Exception:
    requested_portal = ""

if requested_portal == "admin" and not st.session_state.get("authenticated", False):
    st.session_state["login_mode"] = "admin"


st.markdown(
    """
    <style>
    /* =====================================================
       PROFESSIONAL OFFICER AUTHENTICATION UI
       ===================================================== */

    .mplad-auth-page {
        min-height: 78vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 28px 0 50px 0;
    }

    .mplad-auth-brand {
        text-align: center;
        margin-bottom: 22px;
    }

    .mplad-auth-logo {
        width: 58px;
        height: 58px;
        margin: 0 auto 14px auto;
        display: grid;
        place-items: center;
        border-radius: 17px;
        background: linear-gradient(145deg, #2563eb, #4f46e5);
        color: #ffffff;
        font-size: 25px;
        font-weight: 900;
        box-shadow: 0 12px 28px rgba(37,99,235,.28), inset 0 1px 1px rgba(255,255,255,.42);
    }

    .mplad-auth-title {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }

    .mplad-auth-subtitle {
        margin: 7px auto 0 auto;
        max-width: 430px;
        color: #64748b;
        font-size: .88rem;
        line-height: 1.5;
    }

    .mplad-auth-security {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 13px;
    }

    .mplad-auth-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(239,246,255,.85);
        border: 1px solid rgba(37,99,235,.12);
        color: #475569;
        font-size: .70rem;
        font-weight: 700;
    }

    [data-testid="stForm"]:has(.mplad-login-form-marker) {
        padding: 25px 25px 21px 25px !important;
        border: 1px solid rgba(148,163,184,.20) !important;
        border-radius: 22px !important;
        background: rgba(255,255,255,.92) !important;
        box-shadow: 0 24px 60px rgba(15,23,42,.11), inset 0 1px 0 rgba(255,255,255,.95) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .mplad-login-form-marker {
        height: 0;
        overflow: hidden;
        margin: 0;
    }

    .mplad-login-form-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 2px;
    }

    .mplad-login-form-caption {
        color: #94a3b8;
        font-size: .76rem;
        margin-bottom: 13px;
    }

    .mplad-demo-access {
        margin-top: 12px;
        padding: 11px 13px;
        border-radius: 13px;
        border: 1px solid rgba(148,163,184,.18);
        background: rgba(248,250,252,.78);
        color: #64748b;
        font-size: .72rem;
        line-height: 1.6;
    }

    .mplad-demo-access strong {
        color: #334155;
    }

    .mplad-login-footer {
        margin-top: 15px;
        text-align: center;
        color: #94a3b8;
        font-size: .68rem;
        line-height: 1.5;
    }

    .mplad-auth-status {
        margin: 0 auto 14px auto;
        max-width: 560px;
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(239,246,255,.80);
        border: 1px solid rgba(37,99,235,.13);
        color: #475569;
        text-align: center;
        font-size: .80rem;
    }

    @media (max-width: 768px) {
        .mplad-auth-page {
            min-height: auto;
            padding: 12px 0 30px 0;
        }
        .mplad-login-form {
            padding: 20px 17px 17px 17px;
            border-radius: 18px;
        }
        .mplad-auth-title {
            font-size: 1.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


def show_login_page():
    """Render the officer sign-in screen."""

    st.markdown('<div class="mplad-auth-page">', unsafe_allow_html=True)

    left, center, right = st.columns([1.05, 1.35, 1.05])

    with center:
        st.markdown(
            """
            <div class="mplad-auth-brand">
                <div class="mplad-auth-logo">⌁</div>
                <div class="mplad-auth-title">MPLAD-AI</div>
                <div class="mplad-auth-subtitle">
                    Officer access to the MPLAD Risk Intelligence Platform
                </div>
                <div class="mplad-auth-security">
                    <span class="mplad-auth-chip">🔒 Secure access</span>
                    <span class="mplad-auth-chip">✦ AI-assisted review</span>
                </div>
            </div>
            <div class="mplad-login-form-title">Officer Sign In</div>
            <div class="mplad-login-form-caption">
                Sign in with your registered email address or mobile number.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("officer_login_form", clear_on_submit=False):
            login_value = st.text_input(
                "Email or Phone Number",
                placeholder="Enter registered email or 10-digit mobile number",
                key="login_value",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            login_clicked = st.form_submit_button(
                "🔐  Sign In",
                use_container_width=True,
            )

        if login_clicked:
            if not login_value.strip() or not password:
                st.warning("Please enter your email/phone number and password.")
            else:
                officer = authenticate_officer(login_value, password)

                if officer is None:
                    st.error("Invalid email/phone number or password.")
                elif officer.get("not_approved"):
                    if officer.get("approval_status") == "Pending":
                        st.warning(
                            "Your officer account is pending administrator approval. "
                            "You can sign in after the administrator approves your registration."
                        )
                    else:
                        st.error(
                            "Your officer account was rejected. Please contact the administrator."
                        )
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["officer_id"] = officer["id"]
                    st.session_state["officer_name"] = officer["name"]
                    st.session_state["officer_email"] = officer["email"]
                    st.session_state["navigation_section"] = "Dashboard"
                    st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Forgot password?",
                use_container_width=True,
                key="forgot_password_button",
            ):
                st.session_state["password_reset_step"] = "request"
                st.session_state["login_mode"] = "forgot"
                st.rerun()

        with col2:
            if st.button(
                "Create officer account",
                use_container_width=True,
                key="create_account_button",
            ):
                st.session_state["login_mode"] = "register"
                st.rerun()

        st.markdown(
            """
            <div class="mplad-login-footer">
                Authorized officer access only • Passwords are stored using salted secure hashes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def show_register_page():
    """Create a real officer account in the local SQLite database."""

    st.markdown('<div class="mplad-auth-page">', unsafe_allow_html=True)

    left, center, right = st.columns([1.05, 1.35, 1.05])

    with center:
        st.markdown(
            """
            <div class="mplad-auth-brand">
                <div class="mplad-auth-logo">+</div>
                <div class="mplad-auth-title">Officer Registration</div>
                <div class="mplad-auth-subtitle">
                    Create an authenticated MPLAD-AI officer account
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("officer_registration_form", clear_on_submit=False):
            officer_name = st.text_input(
                "Officer Name",
                placeholder="Enter full name",
                key="register_name",
            )
            email = st.text_input(
                "Email Address",
                placeholder="Enter official email address",
                key="register_email",
            )
            phone = st.text_input(
                "Mobile Number",
                placeholder="Enter 10-digit mobile number",
                max_chars=10,
                key="register_phone",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 8 characters",
                key="register_password",
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter password",
                key="register_confirm_password",
            )

            create_clicked = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

        if create_clicked:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                created, message = create_officer_account(
                    officer_name,
                    email,
                    phone,
                    password,
                )

                if created:
                    st.success("Registration submitted successfully. Your account is waiting for administrator approval.")
                    st.session_state["login_mode"] = "login"
                    st.rerun()
                else:
                    st.error(message)

        if st.button(
            "← Back to Sign In",
            use_container_width=True,
            key="back_from_register",
        ):
            st.session_state["login_mode"] = "login"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_admin_page():
    """Administrator dashboard for reviewing officer registrations."""
    st.markdown('<div class="mplad-auth-page">', unsafe_allow_html=True)

    left, center, right = st.columns([0.75, 2.1, 0.75])

    with center:
        st.markdown(
            """
            <div class="mplad-auth-brand">
                <div class="mplad-auth-logo">✓</div>
                <div class="mplad-auth-title">Administrator Portal</div>
                <div class="mplad-auth-subtitle">
                    Review and approve registered MPLAD-AI officer accounts
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.get("admin_authenticated"):
            with st.form("admin_login_form", clear_on_submit=False):
                email = st.text_input(
                    "Administrator Email",
                    placeholder="Enter administrator email",
                )
                password = st.text_input(
                    "Administrator Password",
                    type="password",
                    placeholder="Enter administrator password",
                )
                submitted = st.form_submit_button(
                    "Administrator Sign In",
                    use_container_width=True,
                )

            if submitted:
                if authenticate_admin(email, password):
                    st.session_state["admin_authenticated"] = True
                    st.session_state["admin_email"] = email.strip().lower()
                    st.rerun()
                else:
                    st.error("Invalid administrator credentials.")

            st.info(
                "Administrator credentials are configured through deployment secrets "
                "and are not stored in the officer database."
            )
        else:
            pending = get_pending_officers()
            all_officers = get_all_officers()

            approved_count = sum(1 for row in all_officers if row[5] == "Approved")
            rejected_count = sum(1 for row in all_officers if row[5] == "Rejected")

            m1, m2, m3 = st.columns(3)
            m1.metric("Pending", len(pending))
            m2.metric("Approved", approved_count)
            m3.metric("Rejected", rejected_count)

            st.markdown("### Pending Officer Registrations")

            if not pending:
                st.success("There are no officer registrations waiting for approval.")
            else:
                for row in pending:
                    officer_id, name, email, phone, created_at, status = row

                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1.2])

                        with c1:
                            st.markdown(f"**{name}**")
                            st.caption(
                                f"{email}  •  {phone}  •  Registered: {created_at}"
                            )

                        with c2:
                            approve_col, reject_col = st.columns(2)

                            with approve_col:
                                if st.button(
                                    "Approve",
                                    key=f"approve_officer_{officer_id}",
                                    use_container_width=True,
                                ):
                                    update_officer_approval(
                                        officer_id,
                                        "Approved",
                                        st.session_state["admin_email"],
                                    )
                                    st.rerun()

                            with reject_col:
                                if st.button(
                                    "Reject",
                                    key=f"reject_officer_{officer_id}",
                                    use_container_width=True,
                                ):
                                    update_officer_approval(
                                        officer_id,
                                        "Rejected",
                                        st.session_state["admin_email"],
                                    )
                                    st.rerun()

            st.markdown("### Officer Accounts")

            if all_officers:
                df = pd.DataFrame(
                    all_officers,
                    columns=[
                        "ID", "Officer Name", "Email", "Phone", "Registered",
                        "Status", "Decision Time", "Approved By"
                    ],
                )
                st.dataframe(
                    df[
                        ["ID", "Officer Name", "Email", "Phone", "Registered", "Status"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            if st.button(
                "Logout Administrator",
                use_container_width=True,
                key="admin_logout",
            ):
                st.session_state["admin_authenticated"] = False
                st.session_state["admin_email"] = ""
                st.session_state["login_mode"] = "login"
                st.query_params.clear()
                st.rerun()

        if st.button(
            "← Back to Officer Sign In",
            use_container_width=True,
            key="back_from_admin",
        ):
            st.session_state["login_mode"] = "login"
            st.session_state["admin_authenticated"] = False
            st.session_state["admin_email"] = ""
            st.query_params.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_password_reset_page():
    """Render the real email-OTP password recovery flow."""

    st.markdown('<div class="mplad-auth-page">', unsafe_allow_html=True)

    left, center, right = st.columns([1.05, 1.35, 1.05])

    with center:
        st.markdown(
            """
            <div class="mplad-auth-brand">
                <div class="mplad-auth-logo">↻</div>
                <div class="mplad-auth-title">Reset Password</div>
                <div class="mplad-auth-subtitle">
                    Recover your officer account using an OTP sent to the registered email.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        step = st.session_state.get("password_reset_step", "request")

        if step == "request":
            reset_login = st.text_input(
                "Registered Email or Phone Number",
                placeholder="Enter your registered email or phone",
                key="reset_login",
            )

            if st.button(
                "Send OTP",
                use_container_width=True,
                type="primary",
                key="send_reset_otp",
            ):
                officer = find_officer(reset_login)

                if officer is None:
                    st.error("No officer account was found for that email/phone.")
                else:
                    # OTP recovery is delivered through the officer's registered email.
                    success, message = send_password_reset_otp(officer[2])

                    if success:
                        st.session_state["password_reset_login"] = normalize_login_value(reset_login)
                        st.session_state["password_reset_email"] = officer[2]
                        st.session_state["password_reset_step"] = "otp"
                        st.success(f"OTP sent to {officer[2]}.")
                        st.rerun()
                    else:
                        st.error(message)

            st.caption(
                "The OTP is sent to the registered email address. "
                "No OTP is displayed in the application."
            )

        elif step == "otp":
            st.info(
                f"Enter the 6-digit OTP sent to {st.session_state.get('password_reset_email', '')}."
            )

            otp = st.text_input(
                "Enter OTP",
                max_chars=6,
                placeholder="6-digit OTP",
                key="reset_otp",
            )

            if st.button(
                "Verify OTP",
                use_container_width=True,
                type="primary",
                key="verify_reset_otp",
            ):
                if is_valid_password_reset_otp(otp):
                    st.session_state["password_reset_step"] = "new_password"
                    st.rerun()
                else:
                    st.error("Invalid or expired OTP.")

        elif step == "new_password":
            new_password = st.text_input(
                "New Password",
                type="password",
                placeholder="Minimum 8 characters",
                key="new_reset_password",
            )

            confirm_password = st.text_input(
                "Confirm New Password",
                type="password",
                placeholder="Re-enter password",
                key="confirm_reset_password",
            )

            if st.button(
                "Reset Password",
                use_container_width=True,
                type="primary",
                key="reset_password_button",
            ):
                if len(new_password) < 8:
                    st.error("Password must contain at least 8 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    update_officer_password(
                        st.session_state["password_reset_login"],
                        new_password,
                    )
                    clear_password_reset_state()
                    st.session_state["login_mode"] = "login"
                    st.success("Password reset successfully. You can now sign in.")
                    st.rerun()

        if st.button(
            "← Back to Sign In",
            use_container_width=True,
            key="back_to_login",
        ):
            clear_password_reset_state()
            st.session_state["login_mode"] = "login"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Stop the protected application until an officer logs in.
# ---------------------------------------------------------
if not st.session_state["authenticated"]:
    if st.session_state.get("login_mode") == "forgot":
        show_password_reset_page()
    elif st.session_state.get("login_mode") == "register":
        show_register_page()
    elif st.session_state.get("login_mode") == "admin":
        show_admin_page()
    else:
        show_login_page()

    st.stop()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================


st.sidebar.markdown(
    """
    <div class="mplad-sidebar-brand">
        <div class="mplad-sidebar-orb">⌁</div>
        <div>
            <div class="mplad-sidebar-title">MPLAD-AI</div>
            <div class="mplad-sidebar-subtitle">Risk Intelligence Platform</div>
        </div>
    </div>
    <div class="mplad-sidebar-section">Workspace</div>
    """,
    unsafe_allow_html=True
)

navigation_options = [
    ("Dashboard", "✦  Dashboard"),
    ("Priority Review", "◈  Priority Review"),
    ("Verification", "✓  Verification"),
    ("Check New Work", "+  Check New Work")
]


if "navigation_section" not in st.session_state:
    st.session_state["navigation_section"] = "Dashboard"


def set_navigation(page):
    """Change the active page and let Streamlit rerun once."""
    st.session_state["navigation_section"] = page


# Real Streamlit buttons are used instead of a CSS-modified radio widget.
# This avoids the multi-click issue caused by hiding Streamlit's radio input.
for page_name, button_label in navigation_options:
    is_active = st.session_state["navigation_section"] == page_name

    st.sidebar.button(
        button_label,
        key=f"nav_{page_name.lower().replace(' ', '_')}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        on_click=set_navigation,
        args=(page_name,)
    )


st.sidebar.markdown(
    f"""
    <div class="mplad-sidebar-section" style="margin-top:18px;">Officer</div>
    <div class="mplad-sidebar-brand" style="margin-top:4px;">
        <div class="mplad-sidebar-orb">✓</div>
        <div>
            <div class="mplad-sidebar-title">{st.session_state.get("officer_name", "Officer")}</div>
            <div class="mplad-sidebar-subtitle">{st.session_state.get("officer_email", "")}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button(
    "↪  Logout",
    key="officer_logout",
    use_container_width=True
):
    st.session_state["authenticated"] = False
    st.session_state["officer_name"] = ""
    st.session_state["officer_email"] = ""
    st.session_state.pop("officer_id", None)
    st.session_state["navigation_section"] = "Dashboard"
    st.session_state["login_mode"] = "login"
    st.rerun()


section = st.session_state["navigation_section"]


# =========================================================
# NEW WORK ANALYSIS FUNCTION
# =========================================================

def analyze_new_work(new_work):

    # -----------------------------------------------------
    # 1. Validate state
    # -----------------------------------------------------

    if new_work["STATE_NAME"] not in state_medians.index:

        return {
            "error": "Invalid state. Please select a valid state."
        }


    # -----------------------------------------------------
    # 2. Validate category
    # -----------------------------------------------------

    if new_work["WORK_CATEGORY"] not in category_medians.index:

        return {
            "error": "Invalid work category. Please select a valid category."
        }


    # -----------------------------------------------------
    # 3. Validate amount
    # -----------------------------------------------------

    if new_work["SANCTION_AMOUNT"] <= 0:

        return {
            "error": "Sanction amount must be greater than zero."
        }


    # -----------------------------------------------------
    # 4. Validate delay
    # -----------------------------------------------------

    if new_work["SANCTION_DELAY_DAYS"] < 0:

        return {
            "error": "Sanction delay cannot be negative."
        }


    # -----------------------------------------------------
    # 5. Historical reference values
    # -----------------------------------------------------

    category_median = category_medians[
        new_work["WORK_CATEGORY"]
    ]

    state_median = state_medians[
        new_work["STATE_NAME"]
    ]

    state_delay_median = state_delay_medians[
        new_work["STATE_NAME"]
    ]


    # -----------------------------------------------------
    # 6. Cost deviation
    # -----------------------------------------------------

    cost_deviation = (
        new_work["SANCTION_AMOUNT"]
        / category_median
    )

    state_cost_deviation = (
        new_work["SANCTION_AMOUNT"]
        / state_median
    )


    # -----------------------------------------------------
    # 7. Delay deviation
    # -----------------------------------------------------

    state_delay_deviation = (
        new_work["SANCTION_DELAY_DAYS"]
        / state_delay_median
    )


    # -----------------------------------------------------
    # 8. Log transformations
    # -----------------------------------------------------

    log_cost_deviation = np.log1p(
        cost_deviation
    )

    log_state_cost_deviation = np.log1p(
        state_cost_deviation
    )

    log_sanction_delay = np.log1p(
        new_work["SANCTION_DELAY_DAYS"]
    )

    log_state_delay_deviation = np.log1p(
        state_delay_deviation
    )


    # -----------------------------------------------------
    # 9. Description words
    # -----------------------------------------------------

    description_words = len(
        new_work["WORK_DESCRIPTION"].split()
    )


    # -----------------------------------------------------
    # 10. ML input
    # -----------------------------------------------------

    new_ml_data = pd.DataFrame(
        [[
            log_cost_deviation,
            log_state_cost_deviation,
            log_sanction_delay,
            log_state_delay_deviation,
            description_words
        ]],
        columns=[
            "LOG_COST_DEVIATION",
            "LOG_STATE_COST_DEVIATION",
            "LOG_SANCTION_DELAY",
            "LOG_STATE_DELAY_DEVIATION",
            "DESCRIPTION_WORDS"
        ]
    )


    # -----------------------------------------------------
    # 11. Scale
    # -----------------------------------------------------

    new_ml_scaled = scaler.transform(
        new_ml_data
    )


    # -----------------------------------------------------
    # 12. AI prediction
    # -----------------------------------------------------

    new_prediction = model.predict(
        new_ml_scaled
    )[0]

    new_anomaly_score = model.decision_function(
        new_ml_scaled
    )[0]


    # -----------------------------------------------------
    # 13. Risk percentile calculations
    # -----------------------------------------------------

    cost_risk = (
        work["COST_DEVIATION"]
        <= cost_deviation
    ).mean()

    state_cost_risk = (
        work["STATE_COST_DEVIATION"]
        <= state_cost_deviation
    ).mean()

    delay_risk = (
        work["SANCTION_DELAY_DAYS"]
        <= new_work["SANCTION_DELAY_DAYS"]
    ).mean()

    state_delay_risk = (
        work["STATE_DELAY_DEVIATION"]
        <= state_delay_deviation
    ).mean()


    # -----------------------------------------------------
    # 14. Risk components
    # -----------------------------------------------------

    cost_risk_component = (
        cost_risk + state_cost_risk
    ) / 2

    delay_risk_component = (
        delay_risk + state_delay_risk
    ) / 2


    # -----------------------------------------------------
    # 15. Final risk score
    # -----------------------------------------------------

    base_risk_score = (
        (
            cost_risk_component
            + delay_risk_component
        ) / 2
    ) * 100


    # -----------------------------------------------------
    # 16. Risk level
    # -----------------------------------------------------

    if base_risk_score < 36.14:

        risk_level = "Low"

    elif base_risk_score < 49.86:

        risk_level = "Moderate"

    elif base_risk_score < 63.57:

        risk_level = "Elevated"

    else:

        risk_level = "High"


    # -----------------------------------------------------
    # 17. Reasons
    # -----------------------------------------------------

    reasons = []


    if cost_deviation >= 5:

        reasons.append(
            f"Cost is {cost_deviation:.1f}× "
            "the category median."
        )


    if state_cost_deviation >= 3:

        reasons.append(
            f"Cost is {state_cost_deviation:.1f}× "
            "the state median."
        )


    if new_work["SANCTION_DELAY_DAYS"] >= 300:

        reasons.append(
            f"Sanction delay is "
            f"{new_work['SANCTION_DELAY_DAYS']:.0f} days."
        )


    if state_delay_deviation >= 1.5:

        reasons.append(
            f"Sanction delay is "
            f"{state_delay_deviation:.2f}× "
            "the state median delay."
        )


    if len(reasons) == 0 and new_prediction == -1:

        reasons.append(
            "AI detected an unusual combination "
            "of analyzed patterns."
        )


    # -----------------------------------------------------
    # 18. AI status
    # -----------------------------------------------------

    if new_prediction == -1:

        ai_status = "Anomaly Detected"

    else:

        ai_status = "No Anomaly Detected"


    # -----------------------------------------------------
    # 19. Return
    # -----------------------------------------------------

    return {

        "risk_score":
            round(
                float(base_risk_score),
                2
            ),

        "risk_level":
            risk_level,

        "ai_status":
            ai_status,

        "anomaly_score":
            round(
                float(new_anomaly_score),
                6
            ),

        "cost_deviation":
            round(
                float(cost_deviation),
                2
            ),

        "state_cost_deviation":
            round(
                float(state_cost_deviation),
                2
            ),

        "sanction_delay_days":
            new_work[
                "SANCTION_DELAY_DAYS"
            ],

        "state_delay_deviation":
            round(
                float(state_delay_deviation),
                2
            ),

        "reasons":
            reasons
    }


# =========================================================
# OFFICER REVIEW GUIDANCE
# =========================================================

def get_review_action(risk_level, anomaly_label):
    """
    Returns a simple review action for the officer.

    The action is guidance only. It does not approve, reject,
    or confirm fraud for any work.
    """

    if risk_level == "High" and anomaly_label == -1:
        return (
            "Detailed Review",
            "The work has high relative review priority and "
            "the AI detected an unusual pattern in the analyzed data."
        )

    if risk_level == "High" and anomaly_label == 1:
        return (
            "Review Cost & Delay Justification",
            "The work has high relative review priority based on "
            "cost and delay indicators, although the AI did not "
            "detect an unusual overall pattern."
        )

    if risk_level == "Elevated" and anomaly_label == -1:
        return (
            "Review Unusual Pattern",
            "The AI detected an unusual pattern. Review the "
            "displayed indicators and supporting records."
        )

    if risk_level == "Elevated" and anomaly_label == 1:
        return (
            "Review Cost & Delay Indicators",
            "The work has elevated relative review priority. "
            "Check the displayed cost and delay indicators."
        )

    if risk_level == "Moderate" and anomaly_label == -1:
        return (
            "Check Unusual Pattern",
            "The AI detected an unusual pattern. "
            "Review the available supporting information."
        )

    if risk_level == "Moderate" and anomaly_label == 1:
        return (
            "Routine Review",
            "The work has moderate relative review priority "
            "and no unusual AI pattern was detected."
        )

    if anomaly_label == -1:
        return (
            "Check Unusual Pattern",
            "The AI detected an unusual pattern. "
            "Review the available supporting information."
        )

    return (
        "Routine Review",
        "The work has low relative review priority and "
        "no unusual AI pattern was detected."
    )


# =========================================================
# MPLAD-AI CHATBOT
# =========================================================

def _format_rupees(value):
    value = float(value)
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} crore"
    if value >= 1e5:
        return f"₹{value / 1e5:.2f} lakh"
    return f"₹{value:,.0f}"


def _extract_work_id(question):
    """Extract a Work ID from natural officer wording."""
    q = str(question).strip().lower()

    # Explicit forms: Work 420, Work ID 420, Record No 420, ID: 420, etc.
    explicit_patterns = [
        r"\b(?:work|record)\s*(?:id|no|number)?\s*[:#-]?\s*(\d+)\b",
        r"\b(?:id|work\s*id|record\s*id)\s*[:#-]?\s*(\d+)\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))

    # Natural report requests such as: "give report for 1235"
    # or "1235 report". We only infer a bare number when the
    # question clearly refers to a work/report, avoiding accidental
    # interpretation of ordinary numbers in general questions.
    report_context = any(term in q for term in [
        "report", "summary", "details", "detail", "risk", "delay",
        "cost", "amount", "sanction", "stage", "why", "flag", "work"
    ])
    if report_context:
        number_matches = re.findall(r"\b\d+\b", q)
        if len(number_matches) == 1:
            return int(number_matches[0])

    return None


def _find_work_from_question(question):
    """Find a work using the Work ID used by Priority Review."""
    work_id = _extract_work_id(question)
    if work_id is None:
        return None

    # Priority Review uses the dataframe index as the Work ID.
    if work_id in work.index:
        return work.loc[work_id]

    # Also support the unique recommendation-detail ID.
    matches = work[work["WORK_RECOMMENDATION_DTL_ID"] == work_id]
    if len(matches) > 0:
        return matches.iloc[0]

    return None


def _work_report(selected, work_id):
    """Build a clear officer-friendly report for one work."""
    anomaly = selected["ANOMALY_LABEL"] == -1
    anomaly_text = "Anomaly Detected" if anomaly else "No Anomaly Detected"
    reasons = selected.get("RISK_REASONS", [])
    if not isinstance(reasons, list):
        reasons = []

    reason_text = "\n".join(f"• {reason}" for reason in reasons[:4])
    if not reason_text:
        reason_text = "• No specific rule-based reason was generated."

    recommendation = (
        "Review the supporting records and justification before making an official decision."
        if anomaly or selected["RISK_LEVEL"] in ["Elevated", "High"]
        else "Routine review is appropriate based on the current analyzed indicators."
    )

    return (
        f"### Work {work_id} Report\n\n"
        f"**State:** {selected['STATE_NAME']}  \n"
        f"**Category:** {selected['WORK_CATEGORY']}  \n"
        f"**Stage:** {selected['WORK_STAGE']}  \n"
        f"**Sanction Amount:** {_format_rupees(selected['SANCTION_AMOUNT'])}  \n"
        f"**Recommendation Date:** {selected['RECOMMENDATION_DATE'].strftime('%d %b %Y') if pd.notna(selected['RECOMMENDATION_DATE']) else 'Not available'}  \n"
        f"**Sanction Date:** {selected['SANCTION_DATE'].strftime('%d %b %Y') if pd.notna(selected['SANCTION_DATE']) else 'Not available'}  \n"
        f"**Sanction Delay:** {selected['SANCTION_DELAY_DAYS']:.0f} days  \n\n"
        f"**Risk Level:** {selected['RISK_LEVEL']}  \n"
        f"**Risk Score:** {selected['BASE_RISK_SCORE']:.2f}/100  \n"
        f"**AI Status:** {anomaly_text}  \n\n"
        f"**Why it needs attention:**\n{reason_text}\n\n"
        f"**Officer guidance:** {recommendation}\n\n"
        "*This is an AI screening result for officer review; it does not confirm fraud.*"
    )


def answer_mplad_question(question):
    """Answer officer questions using the loaded MPLAD dataset and risk engine."""
    raw_q = str(question).strip()
    q = raw_q.lower()

    if not q:
        return "Please enter a question about an MPLAD work, risk, delay, state, or anomaly."

    # -----------------------------------------------------
    # Conversational greetings / common officer messages
    # -----------------------------------------------------
    greeting_patterns = [
        r"^hi$", r"^hello$", r"^hey$", r"^hi there$", r"^hello there$",
        r"^good morning$", r"^good afternoon$", r"^good evening$",
        r"^good night$", r"^namaste$"
    ]
    if any(re.fullmatch(pattern, q) for pattern in greeting_patterns):
        if "morning" in q:
            return "Good morning! 👋 I’m MPLAD-AI Assistant. How can I help you with MPLAD works, risks, delays, or verification?"
        if "afternoon" in q:
            return "Good afternoon! 👋 I’m MPLAD-AI Assistant. How can I help you with MPLAD works, risks, delays, or verification?"
        if "evening" in q:
            return "Good evening! 👋 I’m MPLAD-AI Assistant. How can I help you with MPLAD works, risks, delays, or verification?"
        if "night" in q:
            return "Good night! 👋 If you need help with an MPLAD work, risk, delay, or report, I’m here to assist."
        return "Hello! 👋 I’m MPLAD-AI Assistant. How can I help you with MPLAD works, risks, delays, or verification?"

    if any(phrase in q for phrase in ["thank you", "thanks", "thank u", "thx"]):
        return "You’re welcome! I’m here if you need another work report, risk explanation, delay check, or dataset insight."

    if q in {"bye", "goodbye", "see you", "see ya"}:
        return "Goodbye! 👋 I’ll be here whenever you need help reviewing MPLAD works."

    if any(phrase in q for phrase in ["who are you", "what are you", "what can you do", "help me"]):
        return (
            "I’m **MPLAD-AI Assistant**, an officer-support chatbot connected to the analyzed MPLAD dataset. "
            "I can provide work reports, explain risk indicators, check delays and costs, find anomalies by state, "
            "and explain what should be reviewed."
        )

    # -----------------------------------------------------
    # Specific work questions / reports
    # -----------------------------------------------------
    extracted_work_id = _extract_work_id(raw_q)
    selected = _find_work_from_question(raw_q)

    if selected is not None:
        work_id = int(selected.name)

        # "report", "summary", "details", "give report" -> full report.
        if any(term in q for term in ["report", "summary", "details", "give report", "full"]):
            return _work_report(selected, work_id)

        risk_level = selected["RISK_LEVEL"]
        anomaly = selected["ANOMALY_LABEL"] == -1
        anomaly_text = "Anomaly Detected" if anomaly else "No Anomaly Detected"

        reasons = selected.get("RISK_REASONS", [])
        if not isinstance(reasons, list):
            reasons = []

        if "why" in q or "reason" in q or "risk" in q or "flag" in q:
            reason_text = "\n".join(f"• {reason}" for reason in reasons[:4])
            if not reason_text:
                reason_text = "• AI detected no specific rule-based reason for this work."

            return (
                f"**Work {work_id}**\n\n"
                f"**Risk Level:** {risk_level}\n\n"
                f"**Risk Score:** {selected['BASE_RISK_SCORE']:.2f}/100\n\n"
                f"**AI Status:** {anomaly_text}\n\n"
                f"**Reasons:**\n{reason_text}\n\n"
                "These indicators support officer review; they do not confirm fraud."
            )

        if "delay" in q:
            return (
                f"**Work {work_id} sanction delay:** "
                f"{selected['SANCTION_DELAY_DAYS']:.0f} days.\n\n"
                f"State delay deviation: "
                f"{selected['STATE_DELAY_DEVIATION']:.2f}× the state median delay."
            )

        if "amount" in q or "cost" in q:
            return (
                f"**Work {work_id} sanction amount:** "
                f"{_format_rupees(selected['SANCTION_AMOUNT'])}.\n\n"
                f"Category cost deviation: "
                f"{selected['COST_DEVIATION']:.2f}×.\n\n"
                f"State cost deviation: "
                f"{selected['STATE_COST_DEVIATION']:.2f}×."
            )

        return _work_report(selected, work_id)

    # If an officer supplied a work/report ID but it does not exist, say so clearly.
    if extracted_work_id is not None:
        return (
            f"I couldn't find **Work ID {extracted_work_id}** in the current dataset. "
            "Please check the Work ID shown in Priority Review and try again."
        )

    # -----------------------------------------------------
    # Dataset-wide questions
    # -----------------------------------------------------
    total_works = len(work)
    anomaly_count = int((work["ANOMALY_LABEL"] == -1).sum())
    high_count = int((work["RISK_LEVEL"] == "High").sum())
    long_delay_count = int((work["SANCTION_DELAY_DAYS"] >= 300).sum())
    states_count = int(work["STATE_NAME"].nunique())
    total_amount = work["SANCTION_AMOUNT"].sum()

    if any(term in q for term in ["how many works", "total works", "number of works", "works analyzed"]):
        return f"The dataset contains **{total_works:,} works** analyzed by MPLAD-AI."

    if "anomal" in q and any(term in q for term in ["how many", "number", "count", "total"]):
        return (
            f"MPLAD-AI detected **{anomaly_count:,} anomalous works** in the current dataset.\n\n"
            "An anomaly means the work shows an unusual combination of analyzed patterns; it does not confirm fraud."
        )

    if "high priority" in q or "high-risk" in q or "high risk" in q:
        return f"There are **{high_count:,} works** in the dataset with the current **High** relative review-priority level."

    if "long delay" in q or "300 day" in q or "delayed" in q:
        return f"There are **{long_delay_count:,} works** with a sanction delay of 300 days or more."

    if "state" in q and any(term in q for term in ["how many", "number", "covered"]):
        return f"The dataset covers **{states_count} states/UTs** according to the STATE_NAME field."

    if "sanction amount" in q or "total amount" in q or "total sanction" in q:
        return f"The total sanction amount across the analyzed works is **{_format_rupees(total_amount)}**."

    # -----------------------------------------------------
    # State-specific anomaly question
    # -----------------------------------------------------
    for state_name in sorted(work["STATE_NAME"].dropna().unique(), key=len, reverse=True):
        if state_name.lower() in q and "anomal" in q:
            state_data = work[work["STATE_NAME"] == state_name]
            state_anomalies = int((state_data["ANOMALY_LABEL"] == -1).sum())
            return (
                f"**{state_name}** has **{state_anomalies:,} AI-detected anomalous works** "
                f"out of **{len(state_data):,} analyzed works** in the current dataset."
            )

    # -----------------------------------------------------
    # Officer guidance / workflow questions
    # -----------------------------------------------------
    if any(term in q for term in ["how should i review", "how do i review", "review a high-risk", "review high-risk", "what should i verify", "verification checklist"]):
        return (
            "For a **high-risk work**, use the AI result as a screening signal and verify the official record before taking any decision.\n\n"
            "**Recommended review sequence:**\n"
            "1. Verify work description, category and sanction amount.\n"
            "2. Verify recommendation date, sanction date and calculated delay.\n"
            "3. Check the justification for an unusual delay or cost.\n"
            "4. Review sanction orders and supporting documents.\n"
            "5. Check work progress, payment information and completion evidence where available.\n\n"
            "**Important:** An AI anomaly or high risk level does not by itself confirm fraud."
        )

    if "risk score" in q and any(term in q for term in ["mean", "what is", "explain", "understand"]):
        return (
            "The **risk score** is a relative review-priority score from the analyzed dataset. "
            "It combines cost-deviation and sanction-delay indicators. A higher score means the work is "
            "relatively more important to review first; it is **not a probability of fraud** and not an official MPLADS classification."
        )

    if "detect anomal" in q or "how ai detect" in q or "how does ai find" in q:
        return (
            "MPLAD-AI looks for unusual combinations of **cost and sanction-delay patterns**. "
            "It compares a work with category and state-level historical references and also uses an "
            "Isolation Forest anomaly model. The output is an anomaly signal plus explainable indicators for officer review."
        )

    if "high-priority" in q or "high priority" in q or "priority works" in q:
        return (
            f"There are **{high_count:,} high-priority works** under the current dataset-relative risk levels. "
            "You can open **Priority Review** to inspect individual works and start verification."
        )

    # -----------------------------------------------------
    # General explanation
    # -----------------------------------------------------
    if "what is anomaly" in q or "what does anomaly mean" in q:
        return (
            "An anomaly is a work whose analyzed pattern is unusual compared with the "
            "historical patterns in the dataset. It is a screening signal, **not a confirmation of fraud**."
        )

    if "how does" in q and ("risk" in q or "ai" in q):
        return (
            "MPLAD-AI combines cost-deviation and sanction-delay indicators with an "
            "Isolation Forest anomaly model. The result is a relative risk score and "
            "explainable reasons for officer review."
        )

    # -----------------------------------------------------
    # Out-of-scope / unrelated questions
    # -----------------------------------------------------
    # Keep the assistant friendly, but clearly define its role.
    # This prevents unrelated questions from receiving misleading answers.
    return (
        "I’m **MPLAD-AI Assistant**, an officer-support assistant focused on MPLAD works and the current analyzed dataset. "
        "I can help with **work reports, Work IDs, anomaly detection, risk levels, sanction amounts, delays, state-wise analysis, "
        "and verification guidance**.\n\n"
        "I can’t answer unrelated questions outside the MPLAD workflow. "
        "Please ask me something related to an MPLAD work or the risk-analysis dashboard.\n\n"
        "For example, you can ask:\n"
        "• **Give me the report for Work ID 1235**\n"
        "• **Why was this work flagged?**\n"
        "• **How should I review a high-risk work?**\n"
        "• **How many anomalies are there?**\n"
        "• **What does the risk score mean?**"
    )


# =========================================================
# DASHBOARD
# =========================================================


if section == "Dashboard":

    st.title("MPLAD-AI")

    st.subheader(
        "AI-Powered MPLAD Risk Intelligence Platform"
    )

    st.caption(
        "Monitor MPLAD works and identify patterns requiring attention."
    )


    # -----------------------------------------------------
    # Dashboard numbers
    # -----------------------------------------------------

    total_works = len(work)

    total_sanction_amount = work["SANCTION_AMOUNT"].sum()

    anomaly_count = (
        work["ANOMALY_LABEL"] == -1
    ).sum()

    high_count = (
        work["RISK_LEVEL"] == "High"
    ).sum()

    elevated_count = (
        work["RISK_LEVEL"] == "Elevated"
    ).sum()

    important_count = (
        (work["RISK_LEVEL"] == "High")
        | (work["ANOMALY_LABEL"] == -1)
    ).sum()

    moderate_count = (
        work["RISK_LEVEL"] == "Moderate"
    ).sum()

    low_count = (
        work["RISK_LEVEL"] == "Low"
    ).sum()


    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    states_covered = (
        work["STATE_NAME"].nunique()
    )

    st.markdown("## Overview")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Works",
            f"{total_works:,}",
            "Works analyzed"
        )


    with col2:

        st.metric(
            "AI Anomalies",
            f"{anomaly_count:,}",
            "Unusual patterns detected"
        )


    with col3:

        st.metric(
            "High Priority Works",
            f"{high_count:,}",
            "Relative review priority"
        )


    with col4:

        st.metric(
            "States Covered",
            f"{states_covered:,}",
            "States in the dataset"
        )


    # -----------------------------------------------------
    # Risk distribution
    # -----------------------------------------------------

    st.markdown("## Priority Risk Distribution")

    st.caption(
        "Relative review priority based on cost and delay indicators. "
        "These levels are dataset-relative and do not represent confirmed fraud."
    )


    risk_data = {
        "High": high_count,
        "Elevated": elevated_count,
        "Moderate": moderate_count,
        "Low": low_count
    }


    for risk_name, risk_count in risk_data.items():

        percentage = (
            risk_count / total_works
        )

        col1, col2, col3 = st.columns(
            [1, 5, 1]
        )


        with col1:

            st.write(
                f"**{risk_name}**"
            )


        with col2:

            st.progress(
                percentage
            )


        with col3:

            st.write(
                f"**{risk_count:,}**"
            )


    # -----------------------------------------------------
    # AI anomalies by state
    # -----------------------------------------------------

    st.markdown("## AI Anomalies by State")

    state_anomalies = (
        work[work["ANOMALY_LABEL"] == -1]
        .groupby("STATE_NAME")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    st.caption(
        "Top 10 states by number of AI-detected anomalies. "
        "An anomaly indicates an unusual pattern and does not confirm fraud."
    )

    if len(state_anomalies) > 0:

        st.bar_chart(
            state_anomalies,
            horizontal=True
        )

    else:

        st.info(
            "No AI anomalies were detected."
        )


    # -----------------------------------------------------
    # Key Insights
    # -----------------------------------------------------

    st.markdown("## Key Insights")

    long_delay_count = (
        work["SANCTION_DELAY_DAYS"] >= 300
    ).sum()

    insight1, insight2, insight3 = st.columns(3)


    with insight1:

        st.metric(
            "AI Anomalies",
            f"{anomaly_count:,}",
            "Unusual patterns detected"
        )


    with insight2:

        st.metric(
            "Long-Delay Works",
            f"{long_delay_count:,}",
            "300+ days to sanction"
        )


    with insight3:

        st.metric(
            "Total Sanction Amount",
            f"₹{total_sanction_amount:,.0f}",
            "Across all analyzed works"
        )


    st.caption(
        "These indicators support review prioritization "
        "and do not confirm fraud."
    )


    # -----------------------------------------------------
    # How MPLAD-AI helps
    # -----------------------------------------------------

    st.markdown("## How MPLAD-AI Helps")


    help_col1, help_col2, help_col3 = st.columns(3)


    with help_col1:

        st.info(
            "**01 — Analyze**\n\n"
            "Automatically analyzes MPLAD works "
            "using historical patterns."
        )


    with help_col2:

        st.info(
            "**02 — Prioritize**\n\n"
            "Prioritizes works showing unusual "
            "cost or delay patterns."
        )


    with help_col3:

        st.info(
            "**03 — Explain**\n\n"
            "Provides explainable risk indicators "
            "for officer review."
        )


# =========================================================
# PRIORITY REVIEW
# =========================================================

elif section == "Priority Review":

    st.title("Priority Review")

    st.caption(
        "Review works identified by the AI as unusual "
        "and prioritize them using the risk score."
    )


    # -----------------------------------------------------
    # FILTER OPTIONS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        anomaly_filter = st.selectbox(
            "AI Status",
            [
                "Anomaly Detected",
                "All Works",
                "No Anomaly Detected"
            ],
            index=0
        )


    with col2:

        risk_filter = st.selectbox(
            "Risk Level",
            [
                "All",
                "High",
                "Elevated",
                "Moderate",
                "Low"
            ],
            index=0
        )


    # -----------------------------------------------------
    # FILTER DATA
    # -----------------------------------------------------

    priority_works = work.copy()


    if anomaly_filter == "Anomaly Detected":

        priority_works = priority_works[
            priority_works["ANOMALY_LABEL"] == -1
        ]


    elif anomaly_filter == "No Anomaly Detected":

        priority_works = priority_works[
            priority_works["ANOMALY_LABEL"] == 1
        ]


    if risk_filter != "All":

        priority_works = priority_works[
            priority_works["RISK_LEVEL"] == risk_filter
        ]


    priority_works = priority_works.sort_values(
        "BASE_RISK_SCORE",
        ascending=False
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.markdown("## Review Queue")

    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.metric(
            "Works in Queue",
            f"{len(priority_works):,}"
        )


    with summary2:

        queue_anomaly_count = (
            priority_works["ANOMALY_LABEL"] == -1
        ).sum()

        st.metric(
            "AI Anomalies",
            f"{queue_anomaly_count:,}"
        )


    with summary3:

        if len(priority_works) > 0:

            highest_score = priority_works[
                "BASE_RISK_SCORE"
            ].max()

        else:

            highest_score = 0


        st.metric(
            "Highest Risk Score",
            f"{highest_score:.2f}"
        )


    # -----------------------------------------------------
    # WORK TABLE
    # -----------------------------------------------------

    st.markdown("## Works Requiring Review")


    if len(priority_works) > 0:

        display_data = priority_works[
            [
                "STATE_NAME",
                "CONSTITUENCY",
                "SANCTION_AMOUNT",
                "BASE_RISK_SCORE",
                "RISK_LEVEL",
                "AI_STATUS",
                "RISK_REASONS"
            ]
        ].head(25).copy()


        # Work ID comes from the original dataframe index.

        display_data.insert(
            0,
            "WORK_ID",
            display_data.index
        )


        # Show the first explanation as the main reason.

        display_data["MAIN_REASON"] = (
            display_data["RISK_REASONS"]
            .apply(
                lambda reasons:
                reasons[0]
                if isinstance(reasons, list)
                and len(reasons) > 0
                else "AI detected an unusual pattern."
            )
        )


        display_data = display_data.drop(
            columns=["RISK_REASONS"]
        )


        display_data = display_data.rename(
            columns={
                "WORK_ID": "Work ID",
                "STATE_NAME": "State",
                "CONSTITUENCY": "Constituency",
                "SANCTION_AMOUNT": "Sanction Amount",
                "BASE_RISK_SCORE": "Risk Score",
                "RISK_LEVEL": "Risk Level",
                "AI_STATUS": "AI Status",
                "MAIN_REASON": "Main Reason"
            }
        )


        display_data["Sanction Amount"] = (
            display_data["Sanction Amount"]
            .apply(
                lambda x:
                f"₹{x:,.2f}"
            )
        )


        display_data["Risk Score"] = (
            display_data["Risk Score"]
            .apply(
                lambda x:
                f"{x:.2f}"
            )
        )


        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config={

                "Work ID":
                    st.column_config.NumberColumn(
                        "Work ID",
                        width="small"
                    ),

                "State":
                    st.column_config.TextColumn(
                        "State",
                        width="medium"
                    ),

                "Constituency":
                    st.column_config.TextColumn(
                        "Constituency",
                        width="medium"
                    ),

                "Sanction Amount":
                    st.column_config.TextColumn(
                        "Sanction Amount",
                        width="medium"
                    ),

                "Risk Score":
                    st.column_config.TextColumn(
                        "Risk Score",
                        width="small"
                    ),

                "Risk Level":
                    st.column_config.TextColumn(
                        "Risk Level",
                        width="small"
                    ),

                "AI Status":
                    st.column_config.TextColumn(
                        "AI Status",
                        width="medium"
                    ),

                "Main Reason":
                    st.column_config.TextColumn(
                        "Main Reason",
                        width="large"
                    )
            }
        )


        # -------------------------------------------------
        # SEARCH WORK
        # -------------------------------------------------

        st.markdown("## Work Details")
        st.caption("Search a priority work directly by its Work ID.")

        # Store the selected work in session state so the same work remains
        # selected after Streamlit reruns caused by other interactions.
        if "priority_selected_work_id" not in st.session_state:
            st.session_state["priority_selected_work_id"] = int(priority_works.index[0])

        with st.form("priority_work_search_form", clear_on_submit=False):
            search_col1, search_col2 = st.columns([5.5, 1.15])

            with search_col1:
                priority_work_search = st.text_input(
                    "Work ID",
                    value=str(st.session_state["priority_selected_work_id"]),
                    placeholder="Enter Work ID (e.g. 31636)",
                    label_visibility="collapsed",
                    key="priority_work_search_input"
                )

            with search_col2:
                priority_search_button = st.form_submit_button(
                    "🔎 Search",
                    use_container_width=True
                )

        if priority_search_button:
            search_value = priority_work_search.strip()

            if not search_value:
                st.warning("Please enter a Work ID.")
            elif not search_value.isdigit():
                st.warning("Please enter a valid numeric Work ID.")
            else:
                requested_work_id = int(search_value)

                if requested_work_id in priority_works.index:
                    st.session_state["priority_selected_work_id"] = requested_work_id
                else:
                    st.error(
                        f"Work ID {requested_work_id} was not found in the current Priority Review list."
                    )

        selected_index = st.session_state["priority_selected_work_id"]

        # Safety check in case filters change and the previously selected
        # work is no longer present in the filtered Priority Review list.
        if selected_index not in priority_works.index:
            selected_index = int(priority_works.index[0])
            st.session_state["priority_selected_work_id"] = selected_index

        selected_work = priority_works.loc[selected_index]


        # -------------------------------------------------
        # DETAILS
        # -------------------------------------------------

        detail1, detail2, detail3, detail4 = st.columns(
            [1.2, 1.5, 1, 1]
        )


        with detail1:

            st.markdown("**State**")

            st.write(
                selected_work["STATE_NAME"]
            )


        with detail2:

            st.markdown("**Sanction Amount**")

            st.write(
                f"₹{selected_work['SANCTION_AMOUNT']:,.2f}"
            )


        with detail3:

            st.metric(
                "Risk Score",
                f"{selected_work['BASE_RISK_SCORE']:.2f}"
            )


        with detail4:

            st.metric(
                "Risk Level",
                selected_work["RISK_LEVEL"]
            )


        # -------------------------------------------------
        # AI STATUS
        # -------------------------------------------------

        st.markdown("### AI Assessment")


        assessment1, assessment2 = st.columns(2)


        with assessment1:

            st.markdown(
                "#### AI Pattern Status"
            )


            if selected_work["ANOMALY_LABEL"] == -1:

                st.warning(
                    "⚠️ Anomaly Detected"
                )

            else:

                st.success(
                    "✅ No Anomaly Detected"
                )


        with assessment2:

            st.markdown(
                "#### Priority Risk"
            )


            st.info(
                f"Risk Score: "
                f"{selected_work['BASE_RISK_SCORE']:.2f}/100\n\n"
                f"Risk Level: "
                f"{selected_work['RISK_LEVEL']}"
            )


        st.caption(
            "AI Pattern Status identifies unusual patterns "
            "in the analyzed data. Priority Risk represents "
            "relative review priority based on cost and delay "
            "indicators. An anomaly does not mean fraud is confirmed."
        )


        # -------------------------------------------------
        # OFFICER REVIEW GUIDANCE
        # -------------------------------------------------

        review_action, review_message = get_review_action(
            selected_work["RISK_LEVEL"],
            selected_work["ANOMALY_LABEL"]
        )

        st.markdown("### Recommended Review Action")

        st.info(
            f"**{review_action}**\n\n"
            f"{review_message}"
        )

        st.caption(
            "This is review guidance generated from the AI status "
            "and relative priority indicators. The system does not "
            "confirm fraud or make an official decision."
        )


        # -------------------------------------------------
        # REASONS
        # -------------------------------------------------

        if (
            selected_work["RISK_LEVEL"] == "High"
            and selected_work["ANOMALY_LABEL"] == 1
        ):
            reason_heading = (
                "### Why This Work Has High Review Priority"
            )

        elif selected_work["ANOMALY_LABEL"] == -1:
            reason_heading = (
                "### Why This Work Requires Review"
            )

        else:
            reason_heading = (
                "### Assessment Reasons"
            )

        st.markdown(reason_heading)


        reasons = selected_work[
            "RISK_REASONS"
        ]


        if isinstance(reasons, list) and len(reasons) > 0:

            for reason in reasons:

                st.info(reason)

        else:

            st.info(
                "No specific rule-based reason was identified."
            )


    else:

        st.info(
            "No works match the selected filters."
        )


    # -------------------------------------------------
    # OPEN VERIFICATION
    # -------------------------------------------------

    if len(priority_works) > 0:

        st.markdown("### Officer Verification")

        st.caption(
            "After reviewing the AI assessment, open the verification page "
            "to check the selected work against official records and supporting evidence."
        )

        if st.button(
            "Start Verification",
            use_container_width=True
        ):

            st.session_state["verification_work_id"] = int(selected_index)
            st.session_state["verification_work_id_input"] = str(int(selected_index))
            st.session_state["navigation_section"] = "Verification"
            st.rerun()


# =========================================================
# VERIFICATION
# =========================================================

elif section == "Verification":

    st.title("Verification")

    st.caption(
        "Verify the selected work against official records, supporting documents, "
        "photographs and relevant work information. The AI provides review guidance; "
        "the officer makes the final verification decision."
    )

    # -------------------------------------------------
    # SELECT WORK
    # -------------------------------------------------

    default_work_id = st.session_state.get(
        "verification_work_id",
        int(work.sort_values("BASE_RISK_SCORE", ascending=False).index[0])
    )

    # Fast Work ID search: typing does not rerun the whole app.
    # The selected work loads only after the officer presses Search.
    if "verification_work_id_input" not in st.session_state:
        st.session_state["verification_work_id_input"] = str(int(default_work_id))

    st.markdown(
        """
        <div class="verification-search-label">
            <span class="search-dot"></span> Work ID Search
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("verification_work_search", clear_on_submit=False):
        search_col, button_col = st.columns([5.8, 1.2])

        with search_col:
            work_id_text = st.text_input(
                "",
                placeholder="Enter Work ID (e.g. 31636)",
                label_visibility="collapsed",
                key="verification_work_id_input"
            )

        with button_col:
            search_work = st.form_submit_button(
                "🔎 Search",
                use_container_width=True
            )

    if search_work:
        cleaned_work_id = work_id_text.strip()

        if not cleaned_work_id.isdigit():
            st.session_state["verification_search_error"] = (
                "Please enter a valid numeric Work ID."
            )
        else:
            searched_work_id = int(cleaned_work_id)
            st.session_state["verification_work_id"] = searched_work_id
            st.session_state["verification_search_error"] = ""
            st.rerun()

    search_error = st.session_state.get("verification_search_error", "")
    if search_error:
        st.warning(search_error)

    verification_work_id = int(
        st.session_state.get("verification_work_id", default_work_id)
    )

    # Each re-verification gets a fresh widget cycle so the officer
    # starts with a new checklist and review fields.
    verification_cycle = st.session_state.get("verification_cycle", 0)

    if verification_work_id not in work.index:

        st.error("Work ID not found. Please enter a valid Work ID.")

    else:

        selected_work = work.loc[verification_work_id]

        # -------------------------------------------------
        # WORK SUMMARY
        # -------------------------------------------------

        st.markdown("## Work Summary")

        summary1, summary2, summary3, summary4 = st.columns(4)

        with summary1:
            st.markdown("**State**")
            st.write(selected_work["STATE_NAME"])

        with summary2:
            st.markdown("**Constituency**")
            st.write(selected_work["CONSTITUENCY"])

        with summary3:
            st.markdown("**Sanction Amount**")
            st.write(f"₹{selected_work['SANCTION_AMOUNT']:,.2f}")

        with summary4:
            st.markdown("**Work Stage**")
            st.write(selected_work["WORK_STAGE"])

        st.markdown("### AI Assessment")

        ai_col1, ai_col2, ai_col3 = st.columns(3)

        with ai_col1:
            st.metric(
                "Risk Score",
                f"{selected_work['BASE_RISK_SCORE']:.2f}"
            )

        with ai_col2:
            st.metric(
                "Risk Level",
                selected_work["RISK_LEVEL"]
            )

        with ai_col3:
            st.metric(
                "AI Pattern Status",
                "Anomaly Detected"
                if selected_work["ANOMALY_LABEL"] == -1
                else "No Anomaly Detected"
            )

        reasons = selected_work["RISK_REASONS"]

        st.markdown("### Why This Work Was Flagged for Review")

        if isinstance(reasons, list) and len(reasons) > 0:
            for reason in reasons:
                st.info(reason)
        else:
            st.info("No specific rule-based reason was identified.")

        # -------------------------------------------------
        # AI-GUIDED VERIFICATION CHECKLIST
        # -------------------------------------------------

        st.markdown("## AI-Guided Verification Checklist")

        st.caption(
            "Only the checks relevant to this work's detected risk pattern "
            "and current work stage are shown."
        )

        checklist = get_verification_checklist(selected_work)

        st.info(
            f"{len(checklist)} focused verification checks generated for this work. "
            "The officer does not need to verify unrelated items."
        )

        st.markdown(
            "**Verification source:** Use the officer's authorized **official eSAKSHI records** "
            "and available supporting documents/evidence to check each item below."
        )

        st.link_button(
            "Open Official eSAKSHI Portal",
            "https://mplads.mospi.gov.in/digigov/dashboard.html"
        )

        st.caption(
            "MPLAD-AI does not directly verify or modify eSAKSHI records. "
            "It identifies what the officer should check; the officer makes the final verification decision."
        )

        # -------------------------------------------------
        # VERIFICATION RESULT FOR EACH CHECK
        # -------------------------------------------------

        st.markdown("### Verification Results")

        st.caption(
            "For each check, select what the officer found in the official "
            "eSAKSHI record or supporting evidence. A missing document does "
            "not need to be marked as verified."
        )

        verification_options = [
            "Not Checked",
            "Verified",
            "Mismatch / Concern",
            "Not Available / Missing"
        ]

        verification_results = {}

        for number, item in enumerate(checklist, start=1):

            st.markdown(f"**{number}. {item}**")

            result = st.selectbox(
                "Result",
                verification_options,
                key=f"verification_result_{verification_work_id}_{verification_cycle}_{number}",
                label_visibility="collapsed"
            )

            verification_results[item] = result

            st.caption(
                f"Source: {get_verification_source(item)}"
            )

        verified_count = sum(
            value == "Verified"
            for value in verification_results.values()
        )

        concern_count = sum(
            value == "Mismatch / Concern"
            for value in verification_results.values()
        )

        missing_count = sum(
            value == "Not Available / Missing"
            for value in verification_results.values()
        )

        unchecked_count = sum(
            value == "Not Checked"
            for value in verification_results.values()
        )

        st.info(
            f"Verification summary: {verified_count} verified · "
            f"{concern_count} concern(s) · "
            f"{missing_count} missing · "
            f"{unchecked_count} not checked"
        )

        # -------------------------------------------------
        # OFFICER DECISION
        # -------------------------------------------------

        st.markdown("## Officer Review")

        verification_status = st.radio(
            "Verification Status",
            [
                "Pending",
                "Verified - No Issue Found",
                "Requires Further Review",
                "Information / Document Missing",
                "Concern Identified"
            ],
            key=f"verification_status_{verification_work_id}_{verification_cycle}"
        )

        officer_remarks = st.text_area(
            "Officer Remarks",
            placeholder="Enter verification observations or supporting remarks...",
            height=120,
            key=f"verification_remarks_{verification_work_id}_{verification_cycle}"
        )

        if st.button(
            "Submit Verification",
            type="primary",
            use_container_width=True,
            key=f"submit_verification_{verification_work_id}"
        ):

            if verification_status == "Pending":

                st.warning(
                    "Please select a final verification status before submitting."
                )

            elif unchecked_count > 0:

                st.warning(
                    "Please select a result for every verification check before submitting."
                )

            elif (
                verification_status == "Verified - No Issue Found"
                and concern_count > 0
            ):

                st.warning(
                    "The status 'Verified - No Issue Found' cannot be submitted "
                    "while one or more checks show a mismatch or concern."
                )

            elif (
                verification_status == "Verified - No Issue Found"
                and missing_count > 0
            ):

                st.warning(
                    "The status 'Verified - No Issue Found' requires all verification "
                    "checks to be confirmed from available official records."
                )

            elif (
                verification_status == "Concern Identified"
                and concern_count == 0
            ):

                st.warning(
                    "For 'Concern Identified', at least one verification check "
                    "should be marked 'Mismatch / Concern'."
                )

            elif (
                verification_status == "Information / Document Missing"
                and missing_count == 0
            ):

                st.warning(
                    "For 'Information / Document Missing', at least one verification "
                    "check should be marked 'Not Available / Missing'."
                )

            else:

                save_verification(
                    verification_work_id,
                    verification_status,
                    officer_remarks,
                    verification_results
                )

                st.success(
                    f"Verification permanently saved for Work {verification_work_id}."
                )

                st.info(
                    "Saved locally in mplad_verification.db with the work ID, "
                    "checklist, officer status, remarks and submission time."
                )

        st.caption(
            "Verification records are stored in the local SQLite database "
            "mplad_verification.db. This prototype does not update the official eSAKSHI system."
        )

        # -------------------------------------------------
        # VERIFICATION HISTORY
        # -------------------------------------------------

        history = get_verification_history()

        if not history.empty:

            st.markdown("## Verification History")

            st.caption(
                "Each record shows a quick verification summary. "
                "Use View Details to inspect the complete checklist, sources and officer remarks."
            )

            reverify_statuses = {
                "Requires Further Review",
                "Information / Document Missing",
                "Concern Identified"
            }

            st.markdown(
                """
                <style>
                .verification-history-status {
                    display: inline-flex;
                    align-items: center;
                    padding: 5px 10px;
                    border-radius: 999px;
                    font-size: 0.82rem;
                    font-weight: 700;
                    line-height: 1.2;
                    border: 1px solid transparent;
                }
                .verification-status-red {
                    color: #b91c1c;
                    background: #fee2e2;
                    border-color: #fecaca;
                }
                .verification-status-green {
                    color: #166534;
                    background: #dcfce7;
                    border-color: #bbf7d0;
                }
                .verification-status-orange {
                    color: #9a3412;
                    background: #ffedd5;
                    border-color: #fed7aa;
                }
                .verification-summary-line {
                    color: #475569;
                    font-size: 0.86rem;
                    font-weight: 650;
                    margin-top: 4px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Header
            h1, h2, h3, h4, h5, h6 = st.columns(
                [0.6, 0.7, 1.8, 2.2, 1.7, 1.8]
            )
            h1.markdown("**Record**")
            h2.markdown("**Work ID**")
            h3.markdown("**Status**")
            h4.markdown("**Verification Summary**")
            h5.markdown("**Submitted At**")
            h6.markdown("**Action**")

            for _, record in history.iterrows():

                record_id = int(record["id"])
                record_work_id = int(record["work_id"])
                status = str(record["status"])
                submitted_at = str(record["submitted_at"])
                needs_reverification = status in reverify_statuses

                # The checklist is stored as JSON in SQLite.
                # New records store {check item: result}. Older records may
                # contain a list, so keep a safe fallback for compatibility.
                try:
                    saved_checklist = json.loads(record["checklist"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    saved_checklist = {}

                if isinstance(saved_checklist, dict):
                    saved_results = saved_checklist
                elif isinstance(saved_checklist, list):
                    saved_results = {
                        str(item): "Verified" for item in saved_checklist
                    }
                else:
                    saved_results = {}

                verified_total = sum(
                    value == "Verified"
                    for value in saved_results.values()
                )
                concern_total = sum(
                    value == "Mismatch / Concern"
                    for value in saved_results.values()
                )
                missing_total = sum(
                    value == "Not Available / Missing"
                    for value in saved_results.values()
                )
                unchecked_total = sum(
                    value == "Not Checked"
                    for value in saved_results.values()
                )

                if needs_reverification:
                    status_class = "verification-status-red"
                    status_icon = "⚠"
                elif status == "Verified - No Issue Found":
                    status_class = "verification-status-green"
                    status_icon = "✓"
                else:
                    status_class = "verification-status-orange"
                    status_icon = "•"

                c1, c2, c3, c4, c5, c6 = st.columns(
                    [0.6, 0.7, 1.8, 2.2, 1.7, 1.8]
                )

                c1.write(record_id)
                c2.write(record_work_id)

                c3.markdown(
                    f'<span class="verification-history-status {status_class}">'
                    f'{status_icon} {status}'
                    f'</span>',
                    unsafe_allow_html=True
                )

                c4.markdown(
                    f'<div class="verification-summary-line">'
                    f'✅ {verified_total} Verified &nbsp; '
                    f'❌ {concern_total} Concern &nbsp; '
                    f'⚠️ {missing_total} Missing'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if unchecked_total > 0:
                    c4.caption(f"⏳ {unchecked_total} not checked")

                c5.write(submitted_at)

                # Two actions: inspect the saved record or start a fresh
                # verification cycle when the previous result needs review.
                if c6.button(
                    "View Details",
                    key=f"view_verification_{record_id}",
                    use_container_width=True
                ):
                    current_view = st.session_state.get(
                        "verification_detail_record_id"
                    )
                    st.session_state["verification_detail_record_id"] = (
                        None if current_view == record_id else record_id
                    )
                    st.rerun()

                if needs_reverification:
                    if c6.button(
                        "↻ Re-verify",
                        key=f"reverify_{record_id}",
                        use_container_width=True,
                        type="primary"
                    ):
                        st.session_state["verification_work_id"] = record_work_id
                        st.session_state["verification_work_id_input"] = str(record_work_id)
                        st.session_state["verification_cycle"] = verification_cycle + 1
                        st.session_state["verification_detail_record_id"] = None
                        st.rerun()
                else:
                    c6.caption("✓ No re-verification required")

                # Expand only the history record selected with View Details.
                if st.session_state.get("verification_detail_record_id") == record_id:

                    st.markdown(f"### Verification Details — Record {record_id}")

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Work ID", record_work_id)
                    d2.metric("Verified", verified_total)
                    d3.metric("Concerns / Missing", concern_total + missing_total)

                    st.markdown(
                        f'<span class="verification-history-status {status_class}">'
                        f'{status_icon} {status}'
                        f'</span>',
                        unsafe_allow_html=True
                    )

                    st.caption(f"Submitted at: {submitted_at}")

                    st.markdown("#### Saved Checklist Results")

                    if saved_results:
                        for item_number, (item, result) in enumerate(
                            saved_results.items(),
                            start=1
                        ):
                            if result == "Verified":
                                result_icon = "✅"
                            elif result == "Mismatch / Concern":
                                result_icon = "❌"
                            elif result == "Not Available / Missing":
                                result_icon = "⚠️"
                            else:
                                result_icon = "⏳"

                            st.markdown(
                                f"**{item_number}. {item}** — {result_icon} **{result}**"
                            )
                            st.caption(
                                f"Source: {get_verification_source(item)}"
                            )
                    else:
                        st.info(
                            "No structured checklist results are available for this older record."
                        )

                    remarks = str(record["remarks"] or "").strip()
                    st.markdown("#### Officer Remarks")
                    if remarks:
                        st.info(remarks)
                    else:
                        st.caption("No officer remarks were saved for this record.")

                    if st.button(
                        "Close Details",
                        key=f"close_verification_{record_id}"
                    ):
                        st.session_state["verification_detail_record_id"] = None
                        st.rerun()

                st.divider()


# =========================================================
# CHECK NEW WORK
# =========================================================

elif section == "Check New Work":

    st.title("Check New Work")

    st.caption(
        "Enter work details to generate an explainable AI-assisted "
        "risk assessment before official verification."
    )

    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------

    st.markdown(
        """
        <div style="
            margin: 18px 0 12px 0;
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                rgba(37,99,235,0.08),
                rgba(99,102,241,0.06)
            );
            border: 1px solid rgba(37,99,235,0.12);
        ">
            <div style="
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #2563eb;
            ">
                New Work Pre-Screening
            </div>
            <div style="
                margin-top: 5px;
                color: #475569;
                font-size: 0.92rem;
                line-height: 1.5;
            ">
                Compare the entered work with historical MPLADS patterns
                using cost, delay and work-description indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("new_work_form"):

        st.markdown("### Work Information")

        col1, col2 = st.columns(2)

        with col1:

            state = st.selectbox(
                "State",
                sorted(
                    state_medians.index.tolist()
                ),
                help="Select the state used for historical cost and delay comparison."
            )

            category = st.selectbox(
                "Work Category",
                sorted(
                    category_medians.index.tolist()
                ),
                help="Select the category used to calculate the historical category median."
            )

            sanction_amount = st.number_input(
                "Sanction Amount (₹)",
                min_value=1.0,
                value=300000.0,
                step=1000.0,
                help="Enter the proposed or sanctioned amount for the work."
            )

        with col2:

            sanction_delay = st.number_input(
                "Sanction Delay (Days)",
                min_value=0,
                value=0,
                step=1,
                help="Enter the number of days between recommendation and sanction."
            )

            description = st.text_area(
                "Work Description",
                height=140,
                placeholder=(
                    "Example: Construction of a community hall "
                    "with basic public facilities..."
                ),
                help="Provide a clear description of the proposed work."
            )

        analyze_button = st.form_submit_button(
            "🔍 Analyze Work",
            use_container_width=True
        )

    # -----------------------------------------------------
    # RUN ANALYSIS
    # -----------------------------------------------------

    if analyze_button:

        clean_description = description.strip()

        if not clean_description:

            st.error(
                "Please enter a work description before running the analysis."
            )

        elif len(clean_description.split()) < 3:

            st.error(
                "Please provide a more descriptive work description "
                "(at least 3 words)."
            )

        else:

            new_work = {
                "STATE_NAME":
                    state,

                "WORK_CATEGORY":
                    category,

                "SANCTION_AMOUNT":
                    sanction_amount,

                "SANCTION_DELAY_DAYS":
                    sanction_delay,

                "WORK_DESCRIPTION":
                    clean_description
            }

            result = analyze_new_work(
                new_work
            )

            if "error" in result:

                st.error(
                    result["error"]
                )

            else:

                # Store the latest result so it remains visible after
                # normal Streamlit reruns.
                st.session_state["new_work_result"] = result
                st.session_state["new_work_input"] = new_work

    # -----------------------------------------------------
    # DISPLAY LATEST ANALYSIS
    # -----------------------------------------------------

    if "new_work_result" in st.session_state:

        result = st.session_state["new_work_result"]
        analyzed_work = st.session_state["new_work_input"]

        st.success(
            "Work analysis completed successfully."
        )

        st.markdown("## AI Assessment")

        # -------------------------------------------------
        # TOP RESULT CARDS
        # -------------------------------------------------

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Risk Score",
                result["risk_score"]
            )

        with result_col2:

            st.metric(
                "Risk Level",
                result["risk_level"]
            )

        with result_col3:

            st.metric(
                "AI Status",
                result["ai_status"]
            )

        # -------------------------------------------------
        # RISK INTERPRETATION
        # -------------------------------------------------

        risk_level = result["risk_level"]

        if risk_level == "High":
            risk_icon = "🔴"
            risk_message = (
                "High relative review priority. Review the displayed "
                "risk indicators and verify the work against official records."
            )
        elif risk_level == "Elevated":
            risk_icon = "🟠"
            risk_message = (
                "Elevated relative review priority. Review the main "
                "cost and delay indicators."
            )
        elif risk_level == "Moderate":
            risk_icon = "🟡"
            risk_message = (
                "Moderate relative review priority based on the "
                "historical comparison used by the prototype."
            )
        else:
            risk_icon = "🟢"
            risk_message = (
                "Low relative review priority based on the historical "
                "comparison used by the prototype."
            )

        st.markdown(
            f"""
            <div style="
                margin: 12px 0 18px 0;
                padding: 15px 17px;
                border-radius: 15px;
                background: rgba(255,255,255,0.88);
                border: 1px solid rgba(148,163,184,0.20);
                box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            ">
                <div style="
                    font-size: 0.78rem;
                    font-weight: 800;
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                    color: #64748b;
                ">
                    Review Priority
                </div>
                <div style="
                    margin-top: 5px;
                    font-size: 1.02rem;
                    font-weight: 750;
                    color: #0f172a;
                ">
                    {risk_icon} {risk_message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # WHY THIS RESULT
        # -------------------------------------------------

        st.markdown("### Why this result?")

        if result["reasons"]:

            for reason in result["reasons"]:
                st.info(reason)

        else:

            st.info(
                "No specific rule-based risk reason was identified. "
                "The AI model may still identify an unusual combination "
                "of analyzed patterns."
            )

        # -------------------------------------------------
        # HISTORICAL COMPARISON
        # -------------------------------------------------

        st.markdown("### Historical Comparison")

        category_median = float(
            category_medians[
                analyzed_work["WORK_CATEGORY"]
            ]
        )

        state_median = float(
            state_medians[
                analyzed_work["STATE_NAME"]
            ]
        )

        state_delay_median = float(
            state_delay_medians[
                analyzed_work["STATE_NAME"]
            ]
        )

        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

        with comparison_col1:

            st.metric(
                "Entered Amount",
                f"₹{analyzed_work['SANCTION_AMOUNT']:,.0f}"
            )

            st.metric(
                "Category Median",
                f"₹{category_median:,.0f}"
            )

        with comparison_col2:

            st.metric(
                "State Median Amount",
                f"₹{state_median:,.0f}"
            )

            st.metric(
                "Cost vs Category",
                f"{result['cost_deviation']:.2f}×"
            )

        with comparison_col3:

            st.metric(
                "Entered Delay",
                f"{result['sanction_delay_days']} days"
            )

            st.metric(
                "State Median Delay",
                f"{state_delay_median:.0f} days"
            )

        # -------------------------------------------------
        # ANALYZED INDICATORS
        # -------------------------------------------------

        st.markdown("### Analyzed Indicators")

        indicator_col1, indicator_col2 = st.columns(2)

        with indicator_col1:

            st.metric(
                "Category Cost Deviation",
                f"{result['cost_deviation']:.2f}×"
            )

            st.metric(
                "State Cost Deviation",
                f"{result['state_cost_deviation']:.2f}×"
            )

        with indicator_col2:

            st.metric(
                "Sanction Delay",
                f"{result['sanction_delay_days']} days"
            )

            st.metric(
                "State Delay Deviation",
                f"{result['state_delay_deviation']:.2f}×"
            )

        # -------------------------------------------------
        # WORK INPUT SUMMARY
        # -------------------------------------------------

        with st.expander("View Entered Work Details"):

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.write(
                    f"**State:** {analyzed_work['STATE_NAME']}"
                )

                st.write(
                    f"**Work Category:** {analyzed_work['WORK_CATEGORY']}"
                )

                st.write(
                    f"**Sanction Amount:** "
                    f"₹{analyzed_work['SANCTION_AMOUNT']:,.0f}"
                )

            with summary_col2:

                st.write(
                    f"**Sanction Delay:** "
                    f"{analyzed_work['SANCTION_DELAY_DAYS']} days"
                )

                st.write(
                    f"**Description:** "
                    f"{analyzed_work['WORK_DESCRIPTION']}"
                )

        # -------------------------------------------------
        # IMPORTANT AI NOTE
        # -------------------------------------------------

        st.warning(
            "AI screening only: this assessment identifies unusual "
            "patterns relative to the historical dataset. It does not "
            "confirm fraud and does not replace official verification."
        )

        # -------------------------------------------------
        # ACTION
        # -------------------------------------------------

        action_col1, action_col2 = st.columns(2)

        with action_col1:

            if st.button(
                "🔄 Check Another Work",
                use_container_width=True,
                key="reset_new_work"
            ):

                st.session_state.pop(
                    "new_work_result",
                    None
                )

                st.session_state.pop(
                    "new_work_input",
                    None
                )

                st.rerun()

        with action_col2:

            st.caption(
                "Use the Verification workflow for official record checking."
            )

# =========================================================

# FLOATING MPLAD-AI ASSISTANT
# =========================================================

# The assistant is intentionally hidden behind a floating icon.
# Officers can open it from any page without losing the current workflow.

if "mplad_chat_messages" not in st.session_state:
    st.session_state["mplad_chat_messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I’m the **MPLAD-AI Assistant**. Ask me about works, "
                "anomalies, risk levels, delays, costs, or states in the current dataset."
            )
        }
    ]

if "mplad_pending_prompt" not in st.session_state:
    st.session_state["mplad_pending_prompt"] = None

# Sidebar assistant launcher: visible 3D robot card in the navigation empty space.
with st.sidebar.popover("AI", help="Open MPLAD-AI Assistant"):
    st.markdown(
        """
        <div class="mplad-float-chat-head">
            <div class="mplad-float-chat-avatar">✦</div>
            <div>
                <div class="mplad-float-chat-name">MPLAD-AI</div>
                <div class="mplad-float-chat-status">AI Assistant • Dataset connected</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show the latest conversation inside the compact assistant window.
    for message in st.session_state["mplad_chat_messages"][-8:]:
        role = message["role"]
        css_role = "user" if role == "user" else "assistant"
        st.markdown(
            f'<div class="mplad-float-message {css_role}">{message["content"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="mplad-quick-label">Quick questions</div>', unsafe_allow_html=True)

    quick_cols = st.columns(2)
    quick_questions = [
        "How should I review a high-risk work?",
        "How does AI detect anomalies?",
        "How many high-priority works are there?",
        "What does the risk score mean?"
    ]

    for index, question in enumerate(quick_questions):
        with quick_cols[index % 2]:
            if st.button(
                question,
                key=f"mplad_quick_{index}",
                use_container_width=True
            ):
                st.session_state["mplad_pending_prompt"] = question

    chat_prompt = st.text_input(
        "Message",
        value="",
        placeholder="Ask the officer assistant...",
        key="mplad_float_input",
        label_visibility="collapsed"
    )

    send_col, clear_col = st.columns([3, 1])

    with send_col:
        send_clicked = st.button(
            "➤ Send",
            key="mplad_float_send",
            use_container_width=True,
            type="primary"
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear",
            key="mplad_float_clear",
            use_container_width=True
        )

    if clear_clicked:
        st.session_state["mplad_chat_messages"] = [
            {
                "role": "assistant",
                "content": "Chat cleared. How can I help with the MPLAD dataset?"
            }
        ]
        st.session_state["mplad_pending_prompt"] = None
        st.rerun()

    selected_prompt = st.session_state.get("mplad_pending_prompt")
    prompt_to_process = selected_prompt or (chat_prompt if send_clicked else None)

    if prompt_to_process:
        st.session_state["mplad_chat_messages"].append(
            {
                "role": "user",
                "content": prompt_to_process
            }
        )

        answer = answer_mplad_question(prompt_to_process)

        st.session_state["mplad_chat_messages"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.session_state["mplad_pending_prompt"] = None
        st.rerun()

