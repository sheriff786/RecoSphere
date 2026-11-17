"""Services: business rules and transactions. The only layer that uses models. They take a SQLAlchemy Session
and Pydantic schemas, return schemas, and raise ServiceError subclasses (never HTTP errors)."""
