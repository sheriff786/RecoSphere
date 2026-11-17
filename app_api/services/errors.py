"""Service errors. Routers (phase D) map them to HTTP: NotFound 404, Conflict 409, Invalid 422, Unauthorized 401."""


class ServiceError(Exception):
    code = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(ServiceError):
    code = "not_found"


class Conflict(ServiceError):
    code = "conflict"


class Invalid(ServiceError):
    code = "invalid"


class Unauthorized(ServiceError):
    code = "unauthorized"
