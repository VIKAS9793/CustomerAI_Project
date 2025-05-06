# CustomerAI Live Demo Script & Talking Points

## 1. Introduction
- Welcome stakeholders and introduce CustomerAI as a secure, production-ready AI platform for customer insights.
- State the goal: demonstrate real-world readiness, security, and usability.

---

## 2. Secure Login & Authentication
- Show the login screen and enter credentials.
- Point out secure authentication and session management.
- Mention that secrets are validated at startup (show a log or code snippet if desired).

---

## 3. Dashboard Analytics
- Navigate to the main dashboard.
- Highlight real-time analytics, sentiment scores, and fairness checks.
- Show the ability to drill down into customer segments.

---

## 4. API & Error Handling
- Use Postman/cURL to call an API endpoint (e.g., sentiment analysis or fairness check).
- Show what happens if a required secret is missing (simulate or show a log/error message).

---

## 5. CI/CD & Security Enforcement
- Commit a change (or show a sample PR) and demonstrate pre-commit/CI checks in action.
- Intentionally trigger a secret scan or coverage failure to show automated blocking.

---

## 6. Security Features
- Point out Docker hardening: non-root user, read-only filesystem, secure ENV defaults.
- Mention CORS and rate limiting enforced from environment variables.

---

## 7. Q&A and Next Steps
- Invite questions from stakeholders.
- Summarize: “CustomerAI is ready for real-world deployment, with security and quality enforced by automation at every stage.”
