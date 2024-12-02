from __future__ import annotations
import asyncio
from typing import Optional
import rio
from .. import data_models, persistence
from datetime import datetime, timezone


class UserSignUpForm(rio.Component):
    """A component for user registration with validation and security features."""

    # Component state
    popup_open: bool
    _is_processing: bool = False

    # Form fields
    username: str = ""
    email: str = ""
    password: str = ""
    password_confirm: str = ""

    # Validation state
    error_message: str = ""
    field_validity: dict = {"username": True, "email": True, "password": True}

    async def _handle_signup_error(
        self, message: str, field: Optional[str] = None
    ) -> None:
        """Handle signup errors with timing attack prevention"""
        self.error_message = message
        if field:
            self.field_validity[field] = False
        await asyncio.sleep(1)

    async def _validate_form(self) -> bool:
        """Validate form inputs"""
        # Reset validation state
        self.field_validity = {k: True for k in self.field_validity}

        # Check required fields
        if not all([self.username, self.email, self.password, self.password_confirm]):
            await self._handle_signup_error("Please fill in all fields")
            return False

        # Validate email format
        if "@" not in self.email:  # Simple validation, enhance as needed
            await self._handle_signup_error("Invalid email format", "email")
            return False

        # Validate password match
        if self.password != self.password_confirm:
            await self._handle_signup_error("Passwords do not match", "password")
            return False

        # Validate password strength (example)
        if len(self.password) < 8:
            await self._handle_signup_error(
                "Password must be at least 8 characters", "password"
            )
            return False

        return True

    async def on_sign_up(self) -> None:
        """Handle user registration process"""
        if self._is_processing:
            return

        try:
            self._is_processing = True
            await self.force_refresh()

            # Validate form
            if not await self._validate_form():
                return

            pers = self.session[persistence.UserPersistence]

            # Check username availability
            try:
                await pers.get_user(self.username)
                await self._handle_signup_error("Username already taken", "username")
                return
            except KeyError:
                pass

            # Create new user
            user = data_models.AppUser.new_with_defaults(
                username=self.username, email=self.email, password=self.password
            )
            await pers.create_user(
                self.username, password=self.password, email=self.email
            )

            # Create session
            session = await pers.create_session(str(user.id))

            # Update settings
            settings = self.session[data_models.UserSettings]
            settings.auth_token = session.session_id
            settings.last_login = datetime.now(timezone.utc)

            # Attach session data
            self.session.attach(session)
            self.session.attach(user)
            self.session.attach(settings)

            # Close popup and redirect
            self.popup_open = False
            self.session.navigate_to("/app/home")

        finally:
            self._is_processing = False

    def on_cancel(self) -> None:
        """Reset form and close popup"""
        self.username = ""
        self.email = ""
        self.password = ""
        self.password_confirm = ""
        self.error_message = ""
        self.field_validity = {k: True for k in self.field_validity}
        self.popup_open = False

    def build(self) -> rio.Component:
        return rio.Card(
            rio.Column(
                rio.Text("Create Account", style="heading1"),
                rio.Banner(
                    text=self.error_message,
                    style="danger",
                    margin_top=1,
                    # visible=bool(self.error_message)
                ),
                rio.TextInput(
                    text=self.bind().username,
                    label="Username*",
                    is_valid=self.field_validity["username"],
                ),
                rio.TextInput(
                    text=self.bind().email,
                    label="Email*",
                    is_valid=self.field_validity["email"],
                ),
                rio.TextInput(
                    text=self.bind().password,
                    label="Password*",
                    is_valid=self.field_validity["password"],
                    is_secret=True,
                ),
                rio.TextInput(
                    text=self.bind().password_confirm,
                    label="Confirm Password*",
                    is_valid=self.field_validity["password"],
                    is_secret=True,
                ),
                rio.Row(
                    rio.Button(
                        "Sign Up",
                        on_press=self.on_sign_up,
                        is_loading=self._is_processing,
                    ),
                    rio.Button(
                        "Cancel",
                        on_press=self.on_cancel,
                        style="plain-text",
                    ),
                    spacing=2,
                ),
                spacing=1,
                margin=2,
            ),
            align_x=0.5,
            align_y=0.5,
        )
