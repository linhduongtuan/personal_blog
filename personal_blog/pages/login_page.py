from __future__ import annotations
from datetime import datetime, timezone
import rio
import asyncio

from .. import components as comps
from .. import data_models, persistence


class LoginState:
    username: str = ""
    password: str = ""
    error_message: str = ""
    popup_open: bool = False
    _currently_logging_in: bool = False


@rio.page(name="Login", url_segment="")
class LoginPage(rio.Component):
    state: LoginState = rio.field(default_factory=LoginState)

    def _init__(self) -> None:
        super().__init__()
        self.state = LoginState()
        try:
            _ = self.session[persistence.UserPersistence]
        except KeyError:
            self.session.attach(persistence.UserPersistence())

    async def ___init__(self) -> None:
        try:
            _ = self.session[persistence.UserPersistence]
        except KeyError:
            pers = persistence.UserPersistence()
            await pers.__post_init__()
            self.session.attach(pers)

    async def _handle_login_error(self, message: str) -> None:
        self.state.error_message = message
        await asyncio.sleep(1)

    def start_login_process(self) -> None:
        """Synchronous wrapper for async login"""
        asyncio.create_task(self._handle_login())

    async def _handle_login(self) -> None:
        if not self.state.username or not self.state.password:
            await self._handle_login_error("Please fill in all fields")
            return

        try:
            self.state._currently_logging_in = True
            await self.force_refresh()

            pers = self.session[persistence.UserPersistence]

            try:
                user = await pers.get_user_by_username(self.state.username)
            except KeyError:
                await self._handle_login_error("Invalid credentials")
                return

            if await pers.is_account_locked(str(user.id)):
                await self._handle_login_error(
                    "Account locked. Please contact support."
                )
                return

            if not user.verify_password(self.state.password):
                await pers.handle_login_attempt(str(user.id), success=False)
                await self._handle_login_error("Invalid credentials")
                return

            await pers.handle_login_attempt(str(user.id), success=True)
            session = await pers.create_session(str(user.id))

            settings = self.session[data_models.UserSettings]
            settings.auth_token = session.id
            settings.last_login = datetime.now(timezone.utc)

            self.session.attach(session)
            self.session.attach(user)
            self.session.attach(settings)

            self.session.navigate_to("/app/home")

        finally:
            self.state._currently_logging_in = False

    def handle_text_confirm(self, _: rio.TextInputConfirmEvent) -> None:
        """Handle text input confirmation"""
        asyncio.create_task(self._handle_login())

    async def handle_signup_click(self) -> None:
        self.state.popup_open = True
        await self.force_refresh()

    async def handle_popup_close(self, _: rio.GuardEvent) -> None:
        self.state.popup_open = False
        await self.force_refresh()

    def handle_text_change(self, field: str, event: rio.TextInputChangeEvent) -> None:
        setattr(self.state, field, event.text)

    def on_open_popup(self) -> None:
        """
        Opens the sign-up popup when the user clicks the sign-up button
        """
        self.popup_open = True

    def build(self) -> rio.Component:
        column_children: list[rio.Component] = [
            rio.Text("Login", style="heading1"),
        ]

        if self.state.error_message:
            column_children.append(
                rio.Banner(
                    text=self.state.error_message,
                    style="danger",
                    margin_top=1,
                )
            )

        column_children.extend(
            [
                rio.TextInput(
                    text=self.state.username,
                    on_change=lambda e: self.handle_text_change("username", e),
                    label="Username",
                    on_confirm=self.handle_text_confirm,
                ),
                rio.TextInput(
                    text=self.state.password,
                    on_change=lambda e: self.handle_text_change("password", e),
                    label="Password",
                    is_secret=True,
                    on_confirm=self.handle_text_confirm,
                ),
                rio.Row(
                    rio.Button(
                        "Login",
                        on_press=self.start_login_process,
                        is_loading=self.state._currently_logging_in,
                    ),
                    rio.Popup(
                        anchor=rio.Button(
                            "Sign Up",
                            # style="plain-text",
                            on_press=self.handle_signup_click,
                        ),
                        content=comps.UserSignUpForm(
                            popup_open=self.state.popup_open,
                            # on_close=self.handle_popup_close,
                        ),
                        position="fullscreen",
                        is_open=self.state.popup_open,
                        color="none",
                    ),
                    spacing=2,
                ),
            ]
        )

        return rio.Card(
            rio.Column(
                *column_children,
                spacing=1,
                margin=2,
            ),
            align_x=0.5,
            align_y=0.5,
        )
