import os, time
from fastapi import APIRouter, Request, HTTPException
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from jose import jwt


GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
APP_SECRET           = os.getenv("APP_SECRET", "devsecret")   

router = APIRouter(prefix="/auth", tags=["auth"])

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    
    
    client_kwargs={
        "scope": "openid email profile",
        "response_type": "code",   
    },
)

@router.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")  
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request):
    
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not authorize access token: {e}")

    
    id_token = token.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Missing ID token in Google response")

    
    try:
        
        user = jwt.decode(
            id_token,
            key=None,
            algorithms=["RS256"],
            options={
                "verify_signature": False, 
                "verify_at_hash": False   
            },
            audience=GOOGLE_CLIENT_ID
        )
    except jwt.JWTClaimsError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token claims: {e}")

    if not user:
        raise HTTPException(status_code=400, detail="Google login failed")

    user_id = user["sub"]
    email = user["email"]

    
    app_token = jwt.encode(
        {"sub": user_id, "email": email, "exp": int(time.time()) + 3600},
        APP_SECRET,
        algorithm="HS256",
    )

    return RedirectResponse(url=f"https://casacade-ui.vercel.app//chat?token={app_token}")