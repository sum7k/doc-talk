from fastapi import APIRouter

from auth.models.schemas import AuthCreds, MessageResponse, UserCreate, UserOut
from auth.services.auth import AuthHeaderDep, AuthService, AuthServiceDep, CurUserDep

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register")
async def register_user_controller(
    user: UserCreate, auth_service: AuthService = AuthServiceDep
) -> UserOut:
    return await auth_service.register_user(user)


@router.post("/token")
async def login_for_access_token_controller(
    auth_creds: AuthCreds, auth_service=AuthServiceDep
) -> dict:
    access_token = await auth_service.authenticate_user(auth_creds)
    return {"access_token": access_token.access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout_access_token_controller(
    credentials: AuthHeaderDep, auth_service=AuthServiceDep
) -> MessageResponse:
    await auth_service.logout(credentials)
    return MessageResponse(message="Successfully logged out")


@router.get("/me")
async def get_current_user_controller(user: CurUserDep) -> UserOut:
    return user
