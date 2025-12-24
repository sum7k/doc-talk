from typing import Annotated

from fastapi import Depends, HTTPException, status

from auth.models.enums import ROLE_HIERARCHY, UserRole
from auth.models.schemas import UserOut
from auth.services.auth import CurUserDep


def has_min_role(minimum_role: UserRole):
    """User must have at least this role level"""

    async def check(current_user: CurUserDep) -> UserOut:
        user_level = ROLE_HIERARCHY.get(current_user.role, 0)
        min_level = ROLE_HIERARCHY.get(minimum_role, 0)

        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required minimum role: {minimum_role.value}",
            )
        return current_user

    return check


# Predefined dependencies
RequireOwner = Annotated[UserOut, Depends(has_min_role(UserRole.OWNER))]
RequireAdmin = Annotated[UserOut, Depends(has_min_role(UserRole.ADMIN))]
RequireMember = Annotated[UserOut, Depends(has_min_role(UserRole.MEMBER))]
RequireViewer = Annotated[UserOut, Depends(has_min_role(UserRole.VIEWER))]
