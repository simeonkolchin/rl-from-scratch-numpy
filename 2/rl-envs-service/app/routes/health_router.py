from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", summary="Проверить работоспособность")
async def health():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Service is working"})

