# **Contributing to fastapi-skeleton**

Thank you for your interest in contributing to **fastapi-skeleton**!
This project aims to provide a clean, production-ready FastAPI skeleton with **authentication**, **database migrations**, **observability hooks**, and **modern Python patterns**.

Your contributions help improve this ecosystem and make fastapi-skeleton a useful resource for developers building production-ready FastAPI applications.

---

# 🧭 **How to Contribute**

You can contribute in several ways:

### ✔ Report bugs

Open an **Issue** with clear reproduction steps, expected behavior, and your environment details.

### ✔ Improve documentation

Enhance explanations, clarify setup steps, or add examples/tutorials.

### ✔ Suggest new features

Use the **Feature Request** issue type and describe:

* The problem or use-case
* Why it benefits users
* A high-level proposal

### ✔ Submit pull requests (PRs)

Contribute code, tests, or refactoring improvements following the guidelines below.

---

# 🔧 **Project Standards & Expectations**

fastapi-skeleton is meant to be **production-oriented**, so contributions should follow these principles:

### **1. Keep architecture clean and modular**

Code should fit naturally into the existing structure.

* No giant functions
* No untyped dictionaries
* Keep concerns separated (routers ↔ services ↔ repositories ↔ models)

---

### **2. Use type hints everywhere**

We rely on Python’s type system and Pydantic v2 models.

Required:

* Function signatures must be typed
* DB models must be typed
* Service interfaces must be typed

---

### **3. Follow domain-driven design patterns**

When adding or modifying business logic:

* Use Pydantic models for validation
* Enforce validation before database writes
* Keep domain logic separate from infrastructure
* Use repository pattern for data access

---

### **4. Add logs, metrics, or traces if appropriate**

If you add new:

* API endpoints
* Background tasks
* External service calls
* Database operations

Please include:

* **structured logs**
* **appropriate error handling**
* **observability hooks**

This keeps the project reliable and debuggable.

---

### **5. Database changes require Alembic migrations**

If you modify or add tables:

* Update SQLAlchemy/SQLModel models
* Generate Alembic migrations
* Make sure they **upgrade and downgrade cleanly**

PRs that change DB schema without migrations will be rejected.

---

### **6. Keep dependencies minimal**

Avoid unnecessary heavy packages.
Justify new dependencies in your PR description.

---

# 🧪 **Testing Guidelines**

Tests are encouraged for new features:

* Use `pytest` with async support
* Prefer **fast, deterministic tests**
* Use factories for test data (Factory Boy)
* Mock external dependencies
* Test both success and error cases

---

# 📝 **Pull Request Process**

1. Fork the repository & create a feature branch:

   ```bash
   git checkout -b feature/my-awesome-feature
   ```

2. Run formatting tools (if configured):

   ```bash
   black .
   isort .
   ```

3. Ensure the app runs:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Add/update documentation where necessary.

5. Submit a Pull Request with:

   * A clear description (what / why / how)
   * Any caveats or TODOs
   * Screenshots or logs if UI/API behavior changed

6. Participate in code review — be open, calm, and constructive.

---

# 📂 **Good First Issues**

New contributors may enjoy working on:

* Adding more test coverage
* Improving documentation
* Adding observability features
* Enhancing error handling
* Creating diagrams or examples
* Adding new authentication methods
* Implementing additional middleware
* Adding code quality tools

Check the **“good first issue”** label in the Issues tab.

---

# 🤝 **Community Guidelines**

To keep the project healthy:

* Be respectful and constructive
* Keep discussions focused
* Ask if you’re unsure about architectural decisions
* Avoid over-engineering PRs
* Small, incremental improvements are preferred

---

# 🛡️ **License**

fastapi-skeleton is released into the **public domain**.
By submitting a contribution, you agree to release your changes under the same terms.
